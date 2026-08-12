// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

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

use std::ffi::{CString, c_char, c_int, c_uint, c_void};

/// `size_t` for our purposes (HIP's `size_t` is just the C `size_t`).
type c_size_t = usize;
use std::ptr;
use std::sync::Arc;
use std::sync::OnceLock;

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
type FnHipHostMalloc = unsafe extern "C" fn(*mut *mut c_void, c_size_t, c_uint) -> HipError;
type FnHipHostFree = unsafe extern "C" fn(*mut c_void) -> HipError;
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
    /// Optional — used for pinned host I/O when `RLX_ROCM_PINNED_IO=1`.
    pub hip_host_malloc: Option<FnHipHostMalloc>,
    pub hip_host_free: Option<FnHipHostFree>,
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
            macro_rules! sym_opt {
                ($lib:expr, $name:literal, $ty:ty) => {
                    $lib.get($name)
                        .ok()
                        .map(|s: libloading::Symbol<$ty>| *s.into_raw())
                };
            }

            let rt = HipRuntime {
                hip_init: sym!(hip, b"hipInit", FnHipInit),
                hip_get_device_count: sym!(hip, b"hipGetDeviceCount", FnHipGetDeviceCount),
                hip_device_get: sym!(hip, b"hipDeviceGet", FnHipDeviceGet),
                hip_ctx_create: sym!(hip, b"hipCtxCreate", FnHipCtxCreate),
                hip_ctx_destroy: sym!(hip, b"hipCtxDestroy", FnHipCtxDestroy),
                hip_mem_alloc: sym!(hip, b"hipMalloc", FnHipMemAlloc),
                hip_mem_free: sym!(hip, b"hipFree", FnHipMemFree),
                hip_host_malloc: sym_opt!(hip, b"hipHostMalloc", FnHipHostMalloc),
                hip_host_free: sym_opt!(hip, b"hipHostFree", FnHipHostFree),
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
        self.hiprtc_compile_to_hsaco_for(src, name, rocm_target_arch().as_deref())
    }

    /// [`hiprtc_compile_to_hsaco`](Self::hiprtc_compile_to_hsaco) targeting an
    /// explicit arch instead of the resolved one.
    ///
    /// Exists so a failed load can be retried against another candidate: arch
    /// resolution reads `rocm_agent_enumerator` (HSA, *physical* order) while
    /// the code object is loaded onto a HIP device, and those two orderings
    /// need not agree on a mixed-arch box.
    pub fn hiprtc_compile_to_hsaco_for(
        &self,
        src: &str,
        name: &str,
        arch: Option<&str>,
    ) -> Result<Vec<u8>, String> {
        let src_c = CString::new(src).map_err(|e| e.to_string())?;
        let name_c = CString::new(name).map_err(|e| e.to_string())?;

        // Target GPU arch for the emitted code object. With no
        // `--offload-arch`, hipRTC compiles for a default arch whose `.hsaco`
        // won't load on the actual device, surfacing as
        // `hipErrorNoBinaryForGpu` (209) at `hipModuleLoadData` — observed on
        // gfx1103 APUs. Resolve it (env override wins) and pass it explicitly.
        // These backing `CString`s must outlive the compile call below.
        let mut opt_cstrs: Vec<CString> = arch
            .and_then(|a| CString::new(format!("--offload-arch={a}")).ok())
            .into_iter()
            .collect();
        // Opt-in native hardware FP atomics. Without this, HIP lowers
        // `atomicAdd(float*)` to a `global_atomic_cmpswap` retry loop (confirmed by
        // disassembling a kernel — the CAS + `s_cbranch_execnz` back-edge), even on
        // gfx908/CDNA which has `global_atomic_add_f32`. `-munsafe-fp-atomics` emits
        // the native op. FOOTGUN: on fine-grained / PCIe-host memory the hardware FP
        // atomic can silently no-op, so this is gated — rlx's arena is one coarse-
        // grained device allocation where it is safe, but leave the default off until
        // every atomic-using kernel (scatter_add, rms_norm_bwd, …) is validated.
        if rlx_ir::env::flag("RLX_ROCM_FAST_FP_ATOMICS") {
            if let Ok(c) = CString::new("-munsafe-fp-atomics") {
                opt_cstrs.push(c);
            }
        }
        let opt_ptrs: Vec<*const c_char> = opt_cstrs.iter().map(|c| c.as_ptr()).collect();
        let (n_opts, opts_ptr): (c_int, *const *const c_char) = if opt_ptrs.is_empty() {
            (0, ptr::null())
        } else {
            (opt_ptrs.len() as c_int, opt_ptrs.as_ptr())
        };

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

            let compile_status = (self.hiprtc_compile)(prog, n_opts, opts_ptr);
            if compile_status.0 != 0 {
                // Pull the log out before destroying.
                let mut log_size: c_size_t = 0;
                (self.hiprtc_get_log_size)(prog, &mut log_size);
                let mut log = vec![0u8; log_size];
                (self.hiprtc_get_log)(prog, log.as_mut_ptr() as *mut c_char);
                let _ = (self.hiprtc_destroy)(&mut prog);
                // `from_bytes_with_nul` demands EXACTLY one trailing nul, which
                // hipRTC's buffer does not guarantee (it may be unterminated, or
                // padded with several). It therefore failed for every real
                // compile error and printed "<corrupt log>", hiding the actual
                // compiler diagnostic. Cut at the first nul and lossy-decode.
                let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
                let log_str = String::from_utf8_lossy(&log[..end]).into_owned();
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

// ── GPU arch resolution (hipRTC --offload-arch) ──────────────────────

/// Resolve the GPU target arch (e.g. `"gfx1103"`, `"gfx1100"`) used for
/// hipRTC's `--offload-arch`. Resolution order, first hit wins:
///
///   1. `RLX_ROCM_ARCH`            — explicit override (`"gfx1100"`).
///   2. `HSA_OVERRIDE_GFX_VERSION` — the HSA runtime is *reporting* an
///      overridden arch, so the code object must match that, not the
///      physical part (`"11.0.0"` → `"gfx1100"`). This is the common
///      gfx1103/gfx1150 APU workaround.
///   3. `rocm_agent_enumerator` / `rocminfo` — detected physical arch
///      (cached for the process).
///   4. `None` — compile with no `--offload-arch` (legacy behaviour; lets
///      a host whose hipRTC default already matches keep working).
/// The arch a code object has actually been observed to load on this device.
///
/// Set once, the first time a load succeeds after arch resolution guessed
/// wrong. From then on it wins over detection, so exactly one kernel pays the
/// recompile rather than every kernel in the process.
static PROVEN_ARCH: OnceLock<String> = OnceLock::new();

/// Record an arch that `hipModuleLoadData` accepted.
pub(crate) fn note_proven_arch(arch: &str) {
    let _ = PROVEN_ARCH.set(arch.to_string());
}

/// The arch proven to load, if one has been.
pub(crate) fn proven_arch() -> Option<String> {
    PROVEN_ARCH.get().cloned()
}

/// Every GPU arch the ROCm tools report, in physical device order.
///
/// Used as the retry set when the resolved arch is rejected by the device.
pub(crate) fn candidate_archs() -> Vec<String> {
    for cmd in [
        "rocm_agent_enumerator",
        "/opt/rocm/bin/rocm_agent_enumerator",
        "rocminfo",
        "/opt/rocm/bin/rocminfo",
    ] {
        if let Ok(out) = std::process::Command::new(cmd).output() {
            if out.status.success() {
                let archs = all_gfx_tokens(&String::from_utf8_lossy(&out.stdout));
                if !archs.is_empty() {
                    let mut seen = Vec::new();
                    for a in archs {
                        if !seen.contains(&a) {
                            seen.push(a);
                        }
                    }
                    return seen;
                }
            }
        }
    }
    Vec::new()
}

pub fn rocm_target_arch() -> Option<String> {
    // An arch the device has *already accepted* outranks every hint below,
    // including the env overrides. It is only ever set after a successful
    // `hipModuleLoadData`, so it is observed fact rather than inference — and
    // when an override is correct this is simply equal to it. Checking it
    // first is also what makes the retry pay for itself: without this, a
    // process with a wrong `RLX_ROCM_ARCH` would re-run the probe for every
    // single kernel instead of just the first.
    if let Some(a) = proven_arch() {
        return Some(a);
    }
    if let Ok(a) = std::env::var("RLX_ROCM_ARCH") {
        let a = a.trim();
        if !a.is_empty() {
            return Some(a.to_string());
        }
    }
    if let Ok(v) = std::env::var("HSA_OVERRIDE_GFX_VERSION") {
        if let Some(a) = gfx_from_hsa_override(v.trim()) {
            return Some(a);
        }
    }
    detected_rocm_arch().clone()
}

/// Map an `HSA_OVERRIDE_GFX_VERSION` triple (`major.minor.stepping`) to the
/// gfx arch name. The stepping is a single hex nibble in the name, so
/// `11.0.0`→`gfx1100`, `10.3.0`→`gfx1030`, `9.0.10`→`gfx90a`, `9.4.2`→`gfx942`.
fn gfx_from_hsa_override(v: &str) -> Option<String> {
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() != 3 {
        return None;
    }
    let maj: u32 = parts[0].parse().ok()?;
    let min: u32 = parts[1].parse().ok()?;
    let step: u32 = parts[2].parse().ok()?;
    Some(format!("gfx{maj}{min}{step:x}"))
}

/// Physical device arch, detected once via the ROCm CLI tools and cached.
fn detected_rocm_arch() -> &'static Option<String> {
    static ARCH: OnceLock<Option<String>> = OnceLock::new();
    ARCH.get_or_init(|| {
        // `rocm_agent_enumerator` / `rocminfo` list EVERY GPU agent, in physical
        // device order and NOT filtered by `HIP_VISIBLE_DEVICES` (that's a
        // HIP-runtime concept; these are HSA tools). On a multi-GPU box with
        // mixed archs we must select the arch of the device THIS process
        // actually targets — otherwise e.g. a gfx908 code object is force-built
        // (`--offload-arch=gfx908`) and fails to load on a gfx1103 device
        // (hipModuleLoadData → hipError 209).
        let pick = |s: &str| -> Option<String> {
            let archs = all_gfx_tokens(s);
            let idx = visible_device_index().min(archs.len().saturating_sub(1));
            archs.get(idx).cloned()
        };
        for cmd in [
            "rocm_agent_enumerator",
            "/opt/rocm/bin/rocm_agent_enumerator",
            "rocminfo",
            "/opt/rocm/bin/rocminfo",
        ] {
            if let Ok(out) = std::process::Command::new(cmd).output() {
                if out.status.success() {
                    if let Some(a) = pick(&String::from_utf8_lossy(&out.stdout)) {
                        return Some(a);
                    }
                }
            }
        }
        None
    })
}

/// First real GPU arch token in ROCm CLI output — the whitespace-delimited
/// token starting with `gfx` that isn't the `gfx000` CPU/host agent.
/// All GPU gfx arch tokens the enumerator/`rocminfo` reports, in physical
/// device order (CPU agents `gfx000` excluded).
fn all_gfx_tokens(s: &str) -> Vec<String> {
    s.split_whitespace()
        .filter(|t| t.starts_with("gfx") && *t != "gfx000")
        .map(|t| t.to_string())
        .collect()
}

/// The physical GPU index this process targets: the first entry of
/// `HIP_VISIBLE_DEVICES` (or `ROCR_VISIBLE_DEVICES`), which HIP device 0 maps
/// to. Defaults to 0 when unset. Used to select the matching arch from the
/// (unfiltered) agent list so kernels build for the device we actually run on.
fn visible_device_index() -> usize {
    for key in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"] {
        if let Ok(v) = std::env::var(key) {
            if let Some(first) = v.split(',').next() {
                if let Ok(i) = first.trim().parse::<usize>() {
                    return i;
                }
            }
        }
    }
    0
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
    /// Parameters the `__global__` signature declares, parsed from the source
    /// this was compiled from. `None` when it could not be determined.
    pub expected_params: Option<usize>,
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
                expected_params: None,
            })
        }
    }

    /// Attach the parameter count parsed from the source this was compiled
    /// from, enabling the [`Self::launch_checked`] arity check.
    #[must_use]
    pub fn with_expected_params(mut self, n: Option<usize>) -> Self {
        self.expected_params = n;
        self
    }

    /// Launch from a slice, so the argument count is known.
    ///
    /// `hipModuleLaunchKernel` takes a bare `void**` and reads exactly as many
    /// pointers as the compiled kernel declares — it cannot tell how long the
    /// caller's array is. Too few and it reads past the end; too many and the
    /// tail is ignored. Neither faults reliably; the usual symptom is a kernel
    /// that runs and writes nothing. Taking a slice makes the length available,
    /// and with `RLX_GPU_VALIDATE_PARAMS=1` it is checked against the kernel's
    /// own signature before the launch rather than debugged afterwards.
    ///
    /// # Safety
    ///
    /// Same as [`Self::launch`], minus the length obligation: element types
    /// must still match the signature and the pointees outlive the call.
    pub unsafe fn launch_checked(
        &self,
        stream: HipStream,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        kernel_params: &mut [*mut c_void],
    ) -> Result<(), HipError> {
        if let Some(expected) = self.expected_params
            && expected != kernel_params.len()
            && rlx_ir::env::flag("RLX_GPU_VALIDATE_PARAMS")
        {
            panic!(
                "rlx-rocm: launching a kernel with {} argument(s) but its __global__ \
                 signature declares {expected}. hipModuleLaunchKernel reads exactly \
                 {expected} pointers regardless, so this reads past the end of the \
                 argument array (or silently drops the tail) — typically surfacing as a \
                 kernel that completes having written nothing.",
                kernel_params.len()
            );
        }
        unsafe {
            self.launch(
                stream,
                grid,
                block,
                shared_mem_bytes,
                kernel_params.as_mut_ptr(),
            )
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
