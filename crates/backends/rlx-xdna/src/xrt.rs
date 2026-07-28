// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native XRT C-API binding for the XDNA NPU submission path (no Python).
//!
//! The AMD XRT userspace exports a full C ABI from `libxrt_coreutil` (83 `xrt*`
//! symbols) — device open, xclbin load, kernel/run handles, and buffer objects
//! including the **zero-copy** `xrtBOAllocUserPtr`. That lets rlx drive the NPU
//! from Rust with near-zero host overhead, instead of going through `pyxrt`
//! (Python-interpreter cost per dispatch — bad for the ~90 µs-class latencies
//! the NPU runs at).
//!
//! This module is the **binding layer**: it `dlopen`s the runtime and resolves
//! the run-path symbols, so we can prove the native handshake works and hang the
//! executor off it. It does NOT open the device or submit work yet — that needs
//! an AIE-compiled overlay xclbin (see crate docs). Resolving symbols is a pure
//! `dlopen`/`dlsym` (no device I/O, no C++ device init), so it's safe to probe.
//!
//! ## The native low-latency run sequence (what the executor will do)
//!
//! Built ONCE (persistent — the latency-critical amortization):
//!   1. `xrtDeviceOpenByBDF("0000:c6:00.1")` → device handle
//!   2. `xrtDeviceLoadXclbinFile(dev, overlay.xclbin)` → load the INT8 GEMM
//!      overlay; keep the UUID
//!   3. `xrtPLKernelOpen(dev, uuid, "MLIR_AIE")` → kernel handle
//!   4. `xrtBOAllocUserPtr(...)` for **resident weights** + reusable I/O BOs
//!      (zero-copy: map the caller's host buffer, no staging copy)
//!
//! Per call (the hot path — keep it tiny):
//!   5. `xrtBOSync(in_bo, TO_DEVICE)` — only the bytes that changed
//!   6. `xrtRunStart` / `xrtRunSetArg` (reuse one `xrtRunHandle`)
//!   7. `xrtRunWait` (or poll) then `xrtBOSync(out_bo, FROM_DEVICE)`
//!
//! Weights never re-upload; the hwctx/xclbin/kernel/run handles are created once.
//! For the very lowest dispatch overhead, the same sequence can bypass XRT and
//! talk to `/dev/accel*` directly via the `amdxdna` ioctl ABI
//! (`/usr/include/drm/amdxdna_accel.h`) — that's the follow-on path.

use std::ffi::c_void;

/// The run-path symbols we need from `libxrt_coreutil`. Resolved as opaque
/// function pointers — enough to prove the ABI is bindable; the executor casts
/// them to typed signatures when it wires the sequence above.
pub const RUN_PATH_SYMBOLS: &[&[u8]] = &[
    b"xrtDeviceOpenByBDF",
    b"xrtDeviceClose",
    b"xrtDeviceLoadXclbinFile",
    b"xrtPLKernelOpen",
    b"xrtKernelClose",
    b"xrtRunOpen",
    b"xrtRunSetArg",
    b"xrtRunStart",
    b"xrtRunWait",
    b"xrtBOAllocUserPtr",
    b"xrtBOMap",
    b"xrtBOSync",
    b"xrtBOFree",
];

/// A loaded XRT runtime with the run-path symbols resolved.
pub struct XrtBinding {
    // Keep the libs alive for the process; `_core` must outlive `coreutil`
    // (coreutil's `DT_NEEDED` resolves against the already-loaded core).
    _core: Option<libloading::Library>,
    coreutil: libloading::Library,
}

impl XrtBinding {
    /// `dlopen` the XRT runtime from `lib_dir` and confirm every run-path symbol
    /// resolves. `lib_dir` is the directory holding `libxrt_coreutil.so*` (e.g.
    /// `$RLX_XDNA_XRT_LIB` or the extracted `~/xrt-root/.../lib`). Loads the
    /// `libxrt_core` dependency first by absolute path so we don't need the dir
    /// on `LD_LIBRARY_PATH` (which would shadow ROCm's HIP — see crate docs).
    pub fn load(lib_dir: &str) -> Result<Self, String> {
        let so = |stem: &str| first_soname(lib_dir, stem);
        // Best-effort preload of the coreutil dependency chain.
        let core = so("libxrt_core.so")
            .and_then(|p| unsafe { libloading::Library::new(p) }.ok());
        let coreutil_path = so("libxrt_coreutil.so")
            .ok_or_else(|| format!("libxrt_coreutil.so* not found in {lib_dir}"))?;
        let coreutil = unsafe { libloading::Library::new(&coreutil_path) }
            .map_err(|e| format!("dlopen {coreutil_path}: {e}"))?;

        // Resolve each run-path symbol; missing any means an ABI mismatch.
        for sym in RUN_PATH_SYMBOLS {
            unsafe {
                coreutil
                    .get::<unsafe extern "C" fn() -> *mut c_void>(sym)
                    .map_err(|e| {
                        format!(
                            "XRT symbol {} unresolved: {e}",
                            String::from_utf8_lossy(sym)
                        )
                    })?;
            }
        }
        Ok(Self {
            _core: core,
            coreutil,
        })
    }

    /// Number of run-path symbols successfully bound (all of [`RUN_PATH_SYMBOLS`]
    /// once [`load`](Self::load) returns `Ok`).
    pub fn bound_symbols(&self) -> usize {
        RUN_PATH_SYMBOLS
            .iter()
            .filter(|s| unsafe {
                self.coreutil
                    .get::<unsafe extern "C" fn() -> *mut c_void>(s)
                    .is_ok()
            })
            .count()
    }
}

/// `true` if the native XRT C-API can be bound from `lib_dir` (all run-path
/// symbols resolve). A stronger signal than "the `.so` file exists": it proves
/// the ABI this rlx build expects is actually present.
pub fn native_binding_ok(lib_dir: &str) -> bool {
    XrtBinding::load(lib_dir).is_ok()
}

/// First file in `dir` whose name starts with `stem` (matches any soname suffix
/// `.so` / `.so.2` / `.so.2.21.75`), returned as a full path.
fn first_soname(dir: &str, stem: &str) -> Option<String> {
    std::fs::read_dir(dir).ok().and_then(|rd| {
        rd.flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .find(|n| n.starts_with(stem))
            .map(|n| format!("{dir}/{n}"))
    })
}
