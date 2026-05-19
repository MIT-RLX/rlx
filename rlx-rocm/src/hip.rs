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

//! Hand-rolled HIP runtime + hipRTC shim.
//!
//! Same shape as `cudarc::driver` but bounded to the surface we need.
//! Resolves `libamdhip64.so` / `libhiprtc.so` at runtime via libloading
//! so the crate compiles + tests cleanly on hosts without HIP — Mac,
//! Linux without ROCm, CI runners. Returns a clean `None` from
//! `HipRuntime::load()` instead of cudarc's panic-on-missing-driver
//! behaviour (we found the panic-catching probe unhelpful in
//! production logs and decided to be explicit here).
//!
//! ## Coverage
//!
//! Runtime functions wired (everything the kernel-only dispatch path
//! needs):
//!
//!   hipInit, hipGetDeviceCount, hipDeviceGet, hipCtxCreate,
//!   hipCtxDestroy, hipMemAlloc, hipMemFree, hipMemcpyHtoD,
//!   hipMemcpyDtoH, hipMemcpyDtoD, hipModuleLoadData,
//!   hipModuleGetFunction, hipModuleUnload, hipModuleLaunchKernel,
//!   hipStreamCreate, hipStreamSynchronize, hipStreamDestroy,
//!   hipEventCreate, hipEventRecord, hipEventDestroy,
//!   hipStreamWaitEvent, hipDeviceSynchronize.
//!
//! hipRTC functions wired:
//!
//!   hiprtcCreateProgram, hiprtcCompileProgram, hiprtcGetCodeSize,
//!   hiprtcGetCode, hiprtcDestroyProgram, hiprtcGetProgramLogSize,
//!   hiprtcGetProgramLog.
//!
//! Higher-level libraries (hipBLAS / hipBLASLt / MIOpen / hipGraph)
//! are intentionally NOT wired here — they're separate `.so` opens
//! and live in their own modules when the corresponding tiers come
//! online.

#![allow(non_camel_case_types, non_snake_case, dead_code)]

use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};

/// `size_t` for our purposes (HIP's `size_t` is just the C `size_t`).
type c_size_t = usize;
use std::ptr;
use std::sync::Arc;

use libloading::Library;

// ── Opaque pointer types ─────────────────────────────────────────────

/// `hipDevice_t` — really an `int` discriminant in the HIP API; we
/// box it for type safety.
pub type HipDevice = c_int;

/// All HIP "object" types are opaque pointers; we represent them as
/// `*mut c_void` (matching `cudarc`'s representation choice).
pub type HipCtx = *mut c_void;
pub type HipStream = *mut c_void;
pub type HipModule = *mut c_void;
pub type HipFunction = *mut c_void;
pub type HipEvent = *mut c_void;
pub type HipDeviceptr = u64; // `hipDeviceptr_t = void*`; we keep it as u64 for arithmetic
pub type HiprtcProgram = *mut c_void;
pub type HipGraph = *mut c_void;
pub type HipGraphExec = *mut c_void;

/// Error code. 0 = `hipSuccess`; anything else is a failure. We don't
/// decode the variants — the integer value is enough to surface in
/// log_fallback messages.
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HipError(pub c_int);

impl HipError {
    pub fn ok(self) -> Result<(), HipError> {
        if self.0 == 0 { Ok(()) } else { Err(self) }
    }
}

impl std::fmt::Display for HipError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hipError({})", self.0)
    }
}

impl std::error::Error for HipError {}

// ── Function-pointer signatures ──────────────────────────────────────

type FnHipInit = unsafe extern "C" fn(c_uint) -> HipError;
type FnHipGetDeviceCount = unsafe extern "C" fn(*mut c_int) -> HipError;
type FnHipDeviceGet = unsafe extern "C" fn(*mut HipDevice, c_int) -> HipError;
type FnHipCtxCreate = unsafe extern "C" fn(*mut HipCtx, c_uint, HipDevice) -> HipError;
type FnHipCtxDestroy = unsafe extern "C" fn(HipCtx) -> HipError;
type FnHipMemAlloc = unsafe extern "C" fn(*mut HipDeviceptr, c_size_t) -> HipError;
type FnHipMemFree = unsafe extern "C" fn(HipDeviceptr) -> HipError;
type FnHipMemcpyHtoD = unsafe extern "C" fn(HipDeviceptr, *const c_void, c_size_t) -> HipError;
type FnHipMemcpyDtoH = unsafe extern "C" fn(*mut c_void, HipDeviceptr, c_size_t) -> HipError;
type FnHipMemcpyDtoD = unsafe extern "C" fn(HipDeviceptr, HipDeviceptr, c_size_t) -> HipError;
type FnHipModuleLoadData = unsafe extern "C" fn(*mut HipModule, *const c_void) -> HipError;
type FnHipModuleGetFn =
    unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError;
type FnHipModuleUnload = unsafe extern "C" fn(HipModule) -> HipError;
type FnHipLaunchKernel = unsafe extern "C" fn(
    HipFunction,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    c_uint,
    HipStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> HipError;
type FnHipStreamCreate = unsafe extern "C" fn(*mut HipStream) -> HipError;
type FnHipStreamSync = unsafe extern "C" fn(HipStream) -> HipError;
type FnHipStreamDestroy = unsafe extern "C" fn(HipStream) -> HipError;
type FnHipEventCreate = unsafe extern "C" fn(*mut HipEvent, c_uint) -> HipError;
type FnHipEventRecord = unsafe extern "C" fn(HipEvent, HipStream) -> HipError;
type FnHipEventDestroy = unsafe extern "C" fn(HipEvent) -> HipError;
type FnHipStreamWaitEvent = unsafe extern "C" fn(HipStream, HipEvent, c_uint) -> HipError;
type FnHipDeviceSync = unsafe extern "C" fn() -> HipError;
type FnHipStreamBeginCapture = unsafe extern "C" fn(HipStream, c_uint) -> HipError;
type FnHipStreamEndCapture = unsafe extern "C" fn(HipStream, *mut HipGraph) -> HipError;
type FnHipGraphInstantiate = unsafe extern "C" fn(
    *mut HipGraphExec,
    HipGraph,
    *mut c_void,
    *mut c_char,
    c_size_t,
) -> HipError;
type FnHipGraphLaunch = unsafe extern "C" fn(HipGraphExec, HipStream) -> HipError;
type FnHipGraphDestroy = unsafe extern "C" fn(HipGraph) -> HipError;
type FnHipGraphExecDestroy = unsafe extern "C" fn(HipGraphExec) -> HipError;

type FnHiprtcCreate = unsafe extern "C" fn(
    *mut HiprtcProgram,
    *const c_char,
    *const c_char,
    c_int,
    *const *const c_char,
    *const *const c_char,
) -> HipError;
type FnHiprtcCompile = unsafe extern "C" fn(HiprtcProgram, c_int, *const *const c_char) -> HipError;
type FnHiprtcGetCodeSize = unsafe extern "C" fn(HiprtcProgram, *mut c_size_t) -> HipError;
type FnHiprtcGetCode = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HipError;
type FnHiprtcDestroy = unsafe extern "C" fn(*mut HiprtcProgram) -> HipError;
type FnHiprtcGetLogSize = unsafe extern "C" fn(HiprtcProgram, *mut c_size_t) -> HipError;
type FnHiprtcGetLog = unsafe extern "C" fn(HiprtcProgram, *mut c_char) -> HipError;

// ── Loaded runtime ───────────────────────────────────────────────────

/// Resolves all the HIP + hipRTC functions we use up-front. Held in a
/// `OnceLock<Option<Arc<HipRuntime>>>` by `device::rocm_runtime()`.
/// `None` means libamdhip64 / libhiprtc weren't loadable — the crate
/// stays usable but `is_available()` returns false.
pub struct HipRuntime {
    // Keep the libraries alive for the runtime's lifetime.
    _hip_lib: Library,
    _hiprtc_lib: Library,

    // Runtime
    pub hip_init: FnHipInit,
    pub hip_get_device_count: FnHipGetDeviceCount,
    pub hip_device_get: FnHipDeviceGet,
    pub hip_ctx_create: FnHipCtxCreate,
    pub hip_ctx_destroy: FnHipCtxDestroy,
    pub hip_mem_alloc: FnHipMemAlloc,
    pub hip_mem_free: FnHipMemFree,
    pub hip_memcpy_htod: FnHipMemcpyHtoD,
    pub hip_memcpy_dtoh: FnHipMemcpyDtoH,
    pub hip_memcpy_dtod: FnHipMemcpyDtoD,
    pub hip_module_load_data: FnHipModuleLoadData,
    pub hip_module_get_fn: FnHipModuleGetFn,
    pub hip_module_unload: FnHipModuleUnload,
    pub hip_launch_kernel: FnHipLaunchKernel,
    pub hip_stream_create: FnHipStreamCreate,
    pub hip_stream_sync: FnHipStreamSync,
    pub hip_stream_destroy: FnHipStreamDestroy,
    pub hip_event_create: FnHipEventCreate,
    pub hip_event_record: FnHipEventRecord,
    pub hip_event_destroy: FnHipEventDestroy,
    pub hip_stream_wait_event: FnHipStreamWaitEvent,
    pub hip_device_sync: FnHipDeviceSync,
    pub hip_stream_begin_capture: FnHipStreamBeginCapture,
    pub hip_stream_end_capture: FnHipStreamEndCapture,
    pub hip_graph_instantiate: FnHipGraphInstantiate,
    pub hip_graph_launch: FnHipGraphLaunch,
    pub hip_graph_destroy: FnHipGraphDestroy,
    pub hip_graph_exec_destroy: FnHipGraphExecDestroy,

    // hipRTC
    pub hiprtc_create: FnHiprtcCreate,
    pub hiprtc_compile: FnHiprtcCompile,
    pub hiprtc_get_code_size: FnHiprtcGetCodeSize,
    pub hiprtc_get_code: FnHiprtcGetCode,
    pub hiprtc_destroy: FnHiprtcDestroy,
    pub hiprtc_get_log_size: FnHiprtcGetLogSize,
    pub hiprtc_get_log: FnHiprtcGetLog,
}

unsafe impl Send for HipRuntime {}
unsafe impl Sync for HipRuntime {}

impl HipRuntime {
    /// Try to load both `.so`s and resolve every symbol we need.
    /// Returns `None` on any failure (missing library, missing
    /// symbol, etc.).
    pub fn load() -> Option<Arc<Self>> {
        unsafe {
            // Try the canonical Linux paths; libloading falls back to
            // the platform's library search rules.
            let hip = Library::new("libamdhip64.so")
                .or_else(|_| Library::new("libamdhip64.so.6"))
                .or_else(|_| Library::new("libamdhip64.so.5"))
                .ok()?;
            let hiprtc = Library::new("libhiprtc.so")
                .or_else(|_| Library::new("libhiprtc.so.6"))
                .or_else(|_| Library::new("libhiprtc.so.5"))
                .ok()?;

            macro_rules! sym {
                ($lib:expr, $name:literal, $ty:ty) => {{
                    let s: libloading::Symbol<$ty> = $lib.get($name).ok()?;
                    *s.into_raw()
                }};
            }

            let rt = HipRuntime {
                hip_init: sym!(hip, b"hipInit", FnHipInit),
                hip_get_device_count: sym!(hip, b"hipGetDeviceCount", FnHipGetDeviceCount),
                hip_device_get: sym!(hip, b"hipDeviceGet", FnHipDeviceGet),
                hip_ctx_create: sym!(hip, b"hipCtxCreate", FnHipCtxCreate),
                hip_ctx_destroy: sym!(hip, b"hipCtxDestroy", FnHipCtxDestroy),
                hip_mem_alloc: sym!(hip, b"hipMalloc", FnHipMemAlloc),
                hip_mem_free: sym!(hip, b"hipFree", FnHipMemFree),
                hip_memcpy_htod: sym!(hip, b"hipMemcpyHtoD", FnHipMemcpyHtoD),
                hip_memcpy_dtoh: sym!(hip, b"hipMemcpyDtoH", FnHipMemcpyDtoH),
                hip_memcpy_dtod: sym!(hip, b"hipMemcpyDtoD", FnHipMemcpyDtoD),
                hip_module_load_data: sym!(hip, b"hipModuleLoadData", FnHipModuleLoadData),
                hip_module_get_fn: sym!(hip, b"hipModuleGetFunction", FnHipModuleGetFn),
                hip_module_unload: sym!(hip, b"hipModuleUnload", FnHipModuleUnload),
                hip_launch_kernel: sym!(hip, b"hipModuleLaunchKernel", FnHipLaunchKernel),
                hip_stream_create: sym!(hip, b"hipStreamCreate", FnHipStreamCreate),
                hip_stream_sync: sym!(hip, b"hipStreamSynchronize", FnHipStreamSync),
                hip_stream_destroy: sym!(hip, b"hipStreamDestroy", FnHipStreamDestroy),
                hip_event_create: sym!(hip, b"hipEventCreateWithFlags", FnHipEventCreate),
                hip_event_record: sym!(hip, b"hipEventRecord", FnHipEventRecord),
                hip_event_destroy: sym!(hip, b"hipEventDestroy", FnHipEventDestroy),
                hip_stream_wait_event: sym!(hip, b"hipStreamWaitEvent", FnHipStreamWaitEvent),
                hip_device_sync: sym!(hip, b"hipDeviceSynchronize", FnHipDeviceSync),
                hip_stream_begin_capture: sym!(
                    hip,
                    b"hipStreamBeginCapture",
                    FnHipStreamBeginCapture
                ),
                hip_stream_end_capture: sym!(hip, b"hipStreamEndCapture", FnHipStreamEndCapture),
                hip_graph_instantiate: sym!(hip, b"hipGraphInstantiate", FnHipGraphInstantiate),
                hip_graph_launch: sym!(hip, b"hipGraphLaunch", FnHipGraphLaunch),
                hip_graph_destroy: sym!(hip, b"hipGraphDestroy", FnHipGraphDestroy),
                hip_graph_exec_destroy: sym!(hip, b"hipGraphExecDestroy", FnHipGraphExecDestroy),

                hiprtc_create: sym!(hiprtc, b"hiprtcCreateProgram", FnHiprtcCreate),
                hiprtc_compile: sym!(hiprtc, b"hiprtcCompileProgram", FnHiprtcCompile),
                hiprtc_get_code_size: sym!(hiprtc, b"hiprtcGetCodeSize", FnHiprtcGetCodeSize),
                hiprtc_get_code: sym!(hiprtc, b"hiprtcGetCode", FnHiprtcGetCode),
                hiprtc_destroy: sym!(hiprtc, b"hiprtcDestroyProgram", FnHiprtcDestroy),
                hiprtc_get_log_size: sym!(hiprtc, b"hiprtcGetProgramLogSize", FnHiprtcGetLogSize),
                hiprtc_get_log: sym!(hiprtc, b"hiprtcGetProgramLog", FnHiprtcGetLog),

                _hip_lib: hip,
                _hiprtc_lib: hiprtc,
            };
            Some(Arc::new(rt))
        }
    }

    /// Compile a kernel source string to a `.hsaco` binary blob.
    /// Returns `(blob_bytes, log)` on failure so the caller can
    /// surface diagnostics through `log_fallback`.
    pub fn hiprtc_compile_to_hsaco(&self, src: &str, name: &str) -> Result<Vec<u8>, String> {
        let src_c = CString::new(src).map_err(|e| e.to_string())?;
        let name_c = CString::new(name).map_err(|e| e.to_string())?;
        unsafe {
            let mut prog: HiprtcProgram = ptr::null_mut();
            (self.hiprtc_create)(
                &mut prog,
                src_c.as_ptr(),
                name_c.as_ptr(),
                0,
                ptr::null(),
                ptr::null(),
            )
            .ok()
            .map_err(|e| format!("hiprtcCreateProgram: {e}"))?;

            let compile_status = (self.hiprtc_compile)(prog, 0, ptr::null());
            if compile_status.0 != 0 {
                // Pull the log out before destroying.
                let mut log_size: c_size_t = 0;
                (self.hiprtc_get_log_size)(prog, &mut log_size);
                let mut log = vec![0u8; log_size];
                (self.hiprtc_get_log)(prog, log.as_mut_ptr() as *mut c_char);
                let _ = (self.hiprtc_destroy)(&mut prog);
                let log_str = CStr::from_bytes_with_nul(&log)
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|_| "<corrupt log>".to_string());
                return Err(format!("hiprtcCompileProgram: {compile_status}\n{log_str}"));
            }

            let mut code_size: c_size_t = 0;
            (self.hiprtc_get_code_size)(prog, &mut code_size)
                .ok()
                .map_err(|e| format!("hiprtcGetCodeSize: {e}"))?;
            let mut code = vec![0u8; code_size];
            (self.hiprtc_get_code)(prog, code.as_mut_ptr() as *mut c_char)
                .ok()
                .map_err(|e| format!("hiprtcGetCode: {e}"))?;
            let _ = (self.hiprtc_destroy)(&mut prog);
            Ok(code)
        }
    }
}

// ── Convenience wrappers ─────────────────────────────────────────────

/// Owned device buffer. `Drop` releases via `hipFree`.
pub struct HipBuffer<T> {
    rt: Arc<HipRuntime>,
    pub ptr: HipDeviceptr,
    pub len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> HipBuffer<T> {
    pub fn alloc_zeros(rt: &Arc<HipRuntime>, len: usize) -> Result<Self, HipError> {
        unsafe {
            let mut ptr: HipDeviceptr = 0;
            let bytes = len * std::mem::size_of::<T>();
            (rt.hip_mem_alloc)(&mut ptr, bytes).ok()?;
            // Zero via host->device of a zero-filled buffer (cheap +
            // matches rlx-cuda's `alloc_zeros` semantics). hipMemset
            // would be cleaner; saving it for a follow-up.
            let zeros = vec![0u8; bytes];
            (rt.hip_memcpy_htod)(ptr, zeros.as_ptr() as *const c_void, bytes).ok()?;
            Ok(Self {
                rt: rt.clone(),
                ptr,
                len,
                _marker: std::marker::PhantomData,
            })
        }
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> Result<(), HipError> {
        let bytes = std::mem::size_of_val(src);
        unsafe { (self.rt.hip_memcpy_htod)(self.ptr, src.as_ptr() as *const c_void, bytes).ok() }
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<(), HipError> {
        let bytes = std::mem::size_of_val(dst);
        unsafe { (self.rt.hip_memcpy_dtoh)(dst.as_mut_ptr() as *mut c_void, self.ptr, bytes).ok() }
    }
}

impl<T> Drop for HipBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = (self.rt.hip_mem_free)(self.ptr);
            }
        }
    }
}

/// Compiled kernel module + entry function pair (= rlx-cuda's `CudaKernel`).
pub struct HipKernel {
    rt: Arc<HipRuntime>,
    pub module: HipModule,
    pub function: HipFunction,
}

// HIP module + function handles are opaque pointers. The HIP runtime
// is internally thread-safe per its own documentation; mirroring
// cudarc's choice we mark these Send + Sync.
unsafe impl Send for HipKernel {}
unsafe impl Sync for HipKernel {}
unsafe impl<T> Send for HipBuffer<T> {}
unsafe impl<T> Sync for HipBuffer<T> {}

impl Drop for HipKernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = (self.rt.hip_module_unload)(self.module);
            }
        }
    }
}

impl HipKernel {
    pub fn from_hsaco(rt: &Arc<HipRuntime>, hsaco: &[u8], entry: &str) -> Result<Self, HipError> {
        unsafe {
            let mut module: HipModule = ptr::null_mut();
            (rt.hip_module_load_data)(&mut module, hsaco.as_ptr() as *const c_void).ok()?;
            let entry_c = CString::new(entry).expect("entry name has no NULs");
            let mut function: HipFunction = ptr::null_mut();
            (rt.hip_module_get_fn)(&mut function, module, entry_c.as_ptr()).ok()?;
            Ok(Self {
                rt: rt.clone(),
                module,
                function,
            })
        }
    }

    /// Launch with raw `*mut c_void` args, like `cuLaunchKernel`.
    ///
    /// # Safety
    ///
    /// `kernel_params` must point to an array whose length and element
    /// types exactly match the kernel's `__global__` signature; the
    /// pointed-to values must remain valid through the launch (which
    /// hipModuleLaunchKernel returns from synchronously even though
    /// the kernel itself runs asynchronously). Caller must also
    /// ensure `stream` is non-null and belongs to the same context as
    /// `self.module`.
    pub unsafe fn launch(
        &self,
        stream: HipStream,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: *mut *mut c_void,
    ) -> Result<(), HipError> {
        unsafe {
            (self.rt.hip_launch_kernel)(
                self.function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_mem_bytes,
                stream,
                kernel_params,
                ptr::null_mut(),
            )
            .ok()
        }
    }
}
