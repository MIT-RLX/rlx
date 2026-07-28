// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native **INT8 GEMM on the XDNA NPU** — `C[m,n] (i32) = A[m,k] (i8) · B[k,n]
//! (i8)` on the AIE-ML array, driving a precompiled `npu1` MLIR-AIE overlay.
//!
//! The NPU needs XRT's modern `register_xclbin` + `hw_context` flow, which XRT
//! exposes only in C++ (the plain C API's `load_axlf` returns "Operation not
//! supported" on `amdxdna`). So rlx `dlopen`s a tiny C++ shim
//! (`csrc/xrt_gemm_shim.cpp`, compiled against libxrt); the pure-Rust XRT
//! C-API binding in [`crate::xrt`] stays as the zero-dep detection/handshake
//! path, this is the execution path.
//!
//! [`NpuGemm`] is the **persistent** form: [`NpuGemm::open`] does the one-time
//! setup (device open, `register_xclbin`, `hw_context`, kernel, BO alloc,
//! instruction upload — ~150 ms) and [`NpuGemm::run`] is the hot per-call path
//! (sync A/B, run, sync C — a few ms). Reusing one `NpuGemm` across calls is the
//! latency win; a fresh one per call pays cold setup every time. Point it at the
//! compiled shim via `RLX_XDNA_SHIM` (or pass the path).

use std::ffi::{CString, c_char, c_int, c_void};

use crate::XdnaError;

type OpenFn = unsafe extern "C" fn(
    *const c_char, // xclbin path
    *const u32,    // instruction stream (u32 words)
    usize,         // ninstr
    c_int,         // M
    c_int,         // K
    c_int,         // N
) -> *mut c_void; // opaque handle (null on failure)
type RunFn = unsafe extern "C" fn(*mut c_void, *const i8, *const i8, *mut i32) -> c_int;
type SetWeightFn = unsafe extern "C" fn(*mut c_void, *const i8) -> c_int;
type RunAFn = unsafe extern "C" fn(*mut c_void, *const i8, *mut i32) -> c_int;
type SetWeightBlockFn = unsafe extern "C" fn(*mut c_void, c_int, *const i8) -> c_int;
type RunBlockFn = unsafe extern "C" fn(*mut c_void, c_int, *const i8, *mut i32) -> c_int;
type CloseFn = unsafe extern "C" fn(*mut c_void);

fn resolve_shim(shim_path: &str) -> Result<String, XdnaError> {
    if shim_path.is_empty() {
        std::env::var("RLX_XDNA_SHIM")
            .map_err(|_| XdnaError("no shim path (set RLX_XDNA_SHIM or pass shim_path)".into()))
    } else {
        Ok(shim_path.to_string())
    }
}

/// A persistent INT8-GEMM context on the NPU (device/xclbin/hw_context/kernel +
/// resident BOs held open). Create once, [`run`](Self::run) many times; the
/// expensive setup is paid only in [`open`](Self::open).
pub struct NpuGemm {
    // Fields drop in declaration order *after* `Drop::drop` runs, so `close_fn`
    // (called in drop) still sees `_lib` mapped. Keep `_lib` last for clarity.
    handle: *mut c_void,
    run_fn: RunFn,
    set_weight_fn: SetWeightFn,
    run_a_fn: RunAFn,
    set_weight_block_fn: SetWeightBlockFn,
    run_block_fn: RunBlockFn,
    close_fn: CloseFn,
    m: usize,
    k: usize,
    n: usize,
    _lib: libloading::Library, // keep the shim mapped for the fn pointers' life
}

impl NpuGemm {
    /// One-time setup: load the shim, open the NPU, register the overlay, and
    /// allocate the resident buffers. `insts` is the `insts_*.bin` stream (u32
    /// words) paired with `xclbin_path`. `shim_path` empty → `$RLX_XDNA_SHIM`.
    pub fn open(
        shim_path: &str,
        xclbin_path: &str,
        insts: &[u32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self, XdnaError> {
        let shim = resolve_shim(shim_path)?;
        let lib = unsafe { libloading::Library::new(&shim) }
            .map_err(|e| XdnaError(format!("dlopen {shim}: {e}")))?;
        let sym = |name: &[u8]| -> Result<*mut c_void, XdnaError> {
            unsafe { lib.get::<*mut c_void>(name) }
                .map(|s| *s)
                .map_err(|e| {
                    XdnaError(format!(
                        "shim symbol {}: {e}",
                        String::from_utf8_lossy(name)
                    ))
                })
        };
        // Resolve then transmute to typed fn pointers (valid while `lib` lives).
        let open_fn: OpenFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_gemm_open")?) };
        let run_fn: RunFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_gemm_run")?) };
        let set_weight_fn: SetWeightFn =
            unsafe { std::mem::transmute(sym(b"rlx_xdna_gemm_set_weight")?) };
        let run_a_fn: RunAFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_gemm_run_a")?) };
        let set_weight_block_fn: SetWeightBlockFn =
            unsafe { std::mem::transmute(sym(b"rlx_xdna_gemm_set_weight_block")?) };
        let run_block_fn: RunBlockFn =
            unsafe { std::mem::transmute(sym(b"rlx_xdna_gemm_run_block")?) };
        let close_fn: CloseFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_gemm_close")?) };

        let xclbin_c = CString::new(xclbin_path).unwrap();
        let handle = unsafe {
            open_fn(
                xclbin_c.as_ptr(),
                insts.as_ptr(),
                insts.len(),
                m as c_int,
                k as c_int,
                n as c_int,
            )
        };
        if handle.is_null() {
            return Err(XdnaError(
                "rlx_xdna_gemm_open returned null (XRT error on stderr)".into(),
            ));
        }
        Ok(Self {
            handle,
            run_fn,
            set_weight_fn,
            run_a_fn,
            set_weight_block_fn,
            run_block_fn,
            close_fn,
            m,
            k,
            n,
            _lib: lib,
        })
    }

    pub fn dims(&self) -> (usize, usize, usize) {
        (self.m, self.k, self.n)
    }

    /// Upload the weight `B[k,n]` into its resident device buffer **once**;
    /// subsequent [`run_a`](Self::run_a) calls reuse it (skip the B upload). The
    /// LLM-decode shape — fixed weights, streaming activations.
    pub fn set_weight(&self, b: &[i8]) -> Result<(), XdnaError> {
        assert_eq!(b.len(), self.k * self.n, "B must be k*n");
        let rc = unsafe { (self.set_weight_fn)(self.handle, b.as_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!("rlx_xdna_gemm_set_weight returned {rc}")));
        }
        Ok(())
    }

    /// Hot path with a **resident** weight: run `C = A · B` uploading only `A`
    /// (call [`set_weight`](Self::set_weight) first). Returns `[m,n]` i32.
    pub fn run_a(&self, a: &[i8]) -> Result<Vec<i32>, XdnaError> {
        assert_eq!(a.len(), self.m * self.k, "A must be m*k");
        let mut c = vec![0i32; self.m * self.n];
        let rc = unsafe { (self.run_a_fn)(self.handle, a.as_ptr(), c.as_mut_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!("rlx_xdna_gemm_run_a returned {rc}")));
        }
        Ok(c)
    }

    /// Upload weight block `idx` (a full `[k,n]`-shaped overlay tile) into its
    /// own resident device BO — for the tiled path, where each (K-block,N-block)
    /// of the weight stays on-device across [`run_block`](Self::run_block) calls.
    pub fn set_weight_block(&self, idx: usize, b: &[i8]) -> Result<(), XdnaError> {
        assert_eq!(b.len(), self.k * self.n, "weight block must be k*n");
        let rc = unsafe { (self.set_weight_block_fn)(self.handle, idx as c_int, b.as_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!(
                "rlx_xdna_gemm_set_weight_block returned {rc}"
            )));
        }
        Ok(())
    }

    /// Run one tile against resident weight block `idx`, uploading only `A`.
    /// Returns the partial `[m,n]` i32 product (host accumulates over K-blocks).
    pub fn run_block(&self, idx: usize, a: &[i8]) -> Result<Vec<i32>, XdnaError> {
        assert_eq!(a.len(), self.m * self.k, "A block must be m*k");
        let mut c = vec![0i32; self.m * self.n];
        let rc =
            unsafe { (self.run_block_fn)(self.handle, idx as c_int, a.as_ptr(), c.as_mut_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!("rlx_xdna_gemm_run_block returned {rc}")));
        }
        Ok(c)
    }

    /// Hot path: run `C[m,n] i32 = A[m,k] i8 · B[k,n] i8` on the NPU (row-major).
    pub fn run(&self, a: &[i8], b: &[i8]) -> Result<Vec<i32>, XdnaError> {
        assert_eq!(a.len(), self.m * self.k, "A must be m*k");
        assert_eq!(b.len(), self.k * self.n, "B must be k*n");
        let mut c = vec![0i32; self.m * self.n];
        let rc = unsafe { (self.run_fn)(self.handle, a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!(
                "rlx_xdna_gemm_run returned {rc} (XRT error on stderr)"
            )));
        }
        Ok(c)
    }
}

impl Drop for NpuGemm {
    fn drop(&mut self) {
        unsafe { (self.close_fn)(self.handle) }
    }
}

type IoOpenFn = unsafe extern "C" fn(*const c_char, *const u32, usize, c_int) -> *mut c_void;
type IoRunFn = unsafe extern "C" fn(*mut c_void, *const i32, *mut i32) -> c_int;
type IoRun2Fn = unsafe extern "C" fn(*mut c_void, *const i32, *const i32, *mut i32) -> c_int;
type IoCloseFn = unsafe extern "C" fn(*mut c_void);

/// Persistent i32-in / i32-out NPU context — the warm runner for the rlx-emitted
/// elementwise (and other 1-in/1-out) overlays. `open` once, `run` many; the hot
/// path is just sync-in / dispatch / sync-out, so it benches warm like
/// [`NpuGemm`].
pub struct NpuIo {
    handle: *mut c_void,
    run_fn: IoRunFn,
    run2_fn: IoRun2Fn,
    close_fn: IoCloseFn,
    n: usize,
    _lib: libloading::Library,
}

impl NpuIo {
    /// One-time setup for an `n`-element i32 overlay. `shim_path` empty →
    /// `$RLX_XDNA_SHIM`.
    pub fn open(
        shim_path: &str,
        xclbin_path: &str,
        insts: &[u32],
        n: usize,
    ) -> Result<Self, XdnaError> {
        let shim = resolve_shim(shim_path)?;
        let lib = unsafe { libloading::Library::new(&shim) }
            .map_err(|e| XdnaError(format!("dlopen {shim}: {e}")))?;
        let sym = |name: &[u8]| -> Result<*mut c_void, XdnaError> {
            unsafe { lib.get::<*mut c_void>(name) }
                .map(|s| *s)
                .map_err(|e| {
                    XdnaError(format!(
                        "shim symbol {}: {e}",
                        String::from_utf8_lossy(name)
                    ))
                })
        };
        let open_fn: IoOpenFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_io_open")?) };
        let run_fn: IoRunFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_io_run")?) };
        let run2_fn: IoRun2Fn = unsafe { std::mem::transmute(sym(b"rlx_xdna_io_run2")?) };
        let close_fn: IoCloseFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_io_close")?) };
        let xclbin_c = CString::new(xclbin_path).unwrap();
        let handle = unsafe { open_fn(xclbin_c.as_ptr(), insts.as_ptr(), insts.len(), n as c_int) };
        if handle.is_null() {
            return Err(XdnaError(
                "rlx_xdna_io_open returned null (XRT error on stderr)".into(),
            ));
        }
        Ok(Self {
            handle,
            run_fn,
            run2_fn,
            close_fn,
            n,
            _lib: lib,
        })
    }

    /// Hot path: run the overlay over `input` (n i32), returning the n-i32 output.
    pub fn run(&self, input: &[i32]) -> Result<Vec<i32>, XdnaError> {
        assert_eq!(input.len(), self.n, "input must be n");
        let mut out = vec![0i32; self.n];
        let rc = unsafe { (self.run_fn)(self.handle, input.as_ptr(), out.as_mut_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!("rlx_xdna_io_run returned {rc}")));
        }
        Ok(out)
    }

    /// Two-input hot path for **binary** overlays (`out = a ⊙ b`): writes both
    /// input BOs, dispatches, reads the output. Same persistent context as
    /// [`NpuIo::run`]. `a`/`b`/out are all `n` elements of the overlay's dtype
    /// (i32 here; f32/bf16 callers reinterpret the bits, as [`NpuIoF32`] does).
    pub fn run2(&self, a: &[i32], b: &[i32]) -> Result<Vec<i32>, XdnaError> {
        assert_eq!(a.len(), self.n, "a must be n");
        assert_eq!(b.len(), self.n, "b must be n");
        let mut out = vec![0i32; self.n];
        let rc = unsafe { (self.run2_fn)(self.handle, a.as_ptr(), b.as_ptr(), out.as_mut_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!("rlx_xdna_io_run2 returned {rc}")));
        }
        Ok(out)
    }
}

impl Drop for NpuIo {
    fn drop(&mut self) {
        unsafe { (self.close_fn)(self.handle) }
    }
}

// Driven single-threaded through the persistent handle.
unsafe impl Send for NpuIo {}

/// f32 twin of [`NpuIo`]. The XRT buffer objects are just `n*4` bytes, so an f32
/// overlay reuses the **exact same shim I/O path** — we only reinterpret the
/// host f32 bits as i32 across the FFI boundary (same 4-byte cell, no bit
/// movement). This is what lets rlx-emitted f32 activation kernels (e.g.
/// [`crate::aie::emit_relu_f32`]) run on the NPU without a shim rebuild.
pub struct NpuIoF32 {
    inner: NpuIo,
}

impl NpuIoF32 {
    /// One-time setup for an `n`-element f32 overlay. `shim_path` empty →
    /// `$RLX_XDNA_SHIM`.
    pub fn open(
        shim_path: &str,
        xclbin_path: &str,
        insts: &[u32],
        n: usize,
    ) -> Result<Self, XdnaError> {
        Ok(Self {
            inner: NpuIo::open(shim_path, xclbin_path, insts, n)?,
        })
    }

    /// Hot path: run the f32 overlay over `input` (n f32), returning n f32 out.
    /// Reinterprets the f32 slice as i32 (identical 4-byte layout) so the shim's
    /// byte-copy DMA is unchanged; the overlay's MLIR defines the f32 semantics.
    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>, XdnaError> {
        // SAFETY: f32 and i32 share size/alignment; reinterpreting the bit
        // pattern is exactly what the byte-level DMA already does.
        let as_i32: &[i32] =
            unsafe { std::slice::from_raw_parts(input.as_ptr() as *const i32, input.len()) };
        let out_i32 = self.inner.run(as_i32)?;
        Ok(out_i32.iter().map(|&b| f32::from_bits(b as u32)).collect())
    }
}

// Driven single-threaded through the persistent handle.
unsafe impl Send for NpuIoF32 {}

/// Round-to-nearest-even f32 → bf16 (bf16 = the top 16 bits of f32 with proper
/// rounding). This is the host-side cast the NPU's native bf16 activation path
/// uses; the NPU vector FPU then runs 32-wide on the bf16 stream.
pub fn f32_to_bf16(f: f32) -> u16 {
    let bits = f.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return ((bits >> 16) as u16) | 0x0040; // NaN → quiet NaN
    }
    let rounding_bias = 0x0000_7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// bf16 → f32 (widen: bf16 bits become the top 16 bits of the f32).
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// bf16 twin of [`NpuIoF32`] — the **native fast float activation path** on AIE2.
/// The AIE2 vector FPU is bf16 (f32 vector ops are unsupported), so a vectorized
/// float kernel (e.g. [`crate::aie::emit_relu_bf16`]) runs in bf16. The host
/// casts f32↔bf16 at the boundary; the two bf16 cells pack into one i32 cell, so
/// this reuses the exact same byte-level shim I/O path (opened at `n/2` i32).
/// `n` must be even.
pub struct NpuIoBf16 {
    inner: NpuIo,
    n: usize,
}

impl NpuIoBf16 {
    /// One-time setup for an `n`-element bf16 overlay (`n` even). The underlying
    /// i32 shim BO is `n/2` cells = `n*2` bytes = `n` bf16.
    pub fn open(
        shim_path: &str,
        xclbin_path: &str,
        insts: &[u32],
        n: usize,
    ) -> Result<Self, XdnaError> {
        assert_eq!(
            n % 2,
            0,
            "NpuIoBf16 requires an even element count (got {n})"
        );
        Ok(Self {
            inner: NpuIo::open(shim_path, xclbin_path, insts, n / 2)?,
            n,
        })
    }

    /// Hot path: cast f32→bf16, run the bf16 overlay, cast bf16→f32. The bf16
    /// bytes are reinterpreted as i32 for the shim's byte-copy DMA (no bit
    /// movement beyond the cast); the overlay's MLIR defines the bf16 semantics.
    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>, XdnaError> {
        assert_eq!(input.len(), self.n, "input must be n bf16 elements");
        let bf: Vec<u16> = input.iter().map(|&f| f32_to_bf16(f)).collect();
        // SAFETY: 2 u16 (4 bytes) reinterpreted as 1 i32 — same layout; the DMA
        // is byte-level and little-endian preserves element order.
        let as_i32: &[i32] =
            unsafe { std::slice::from_raw_parts(bf.as_ptr() as *const i32, self.n / 2) };
        let out_i32 = self.inner.run(as_i32)?;
        let out_bf: &[u16] =
            unsafe { std::slice::from_raw_parts(out_i32.as_ptr() as *const u16, self.n) };
        Ok(out_bf.iter().map(|&b| bf16_to_f32(b)).collect())
    }
}

// Driven single-threaded through the persistent handle.
unsafe impl Send for NpuIoBf16 {}

type Mm32OpenFn =
    unsafe extern "C" fn(*const c_char, *const u32, usize, c_int, c_int, c_int) -> *mut c_void;
type Mm32RunFn = unsafe extern "C" fn(*mut c_void, *const i32, *const i32, *mut i32) -> c_int;
type Mm32CloseFn = unsafe extern "C" fn(*mut c_void);

/// Persistent i32 `C[m,n] = A[m,k]·B[k,n]` NPU context — the warm runner for the
/// rlx-emitted i32 matmul overlays (all operands i32, vs [`NpuGemm`]'s i8 A/B).
pub struct NpuMm32 {
    handle: *mut c_void,
    run_fn: Mm32RunFn,
    close_fn: Mm32CloseFn,
    m: usize,
    k: usize,
    n: usize,
    _lib: libloading::Library,
}

impl NpuMm32 {
    pub fn open(
        shim_path: &str,
        xclbin_path: &str,
        insts: &[u32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Self, XdnaError> {
        let shim = resolve_shim(shim_path)?;
        let lib = unsafe { libloading::Library::new(&shim) }
            .map_err(|e| XdnaError(format!("dlopen {shim}: {e}")))?;
        let sym = |name: &[u8]| -> Result<*mut c_void, XdnaError> {
            unsafe { lib.get::<*mut c_void>(name) }
                .map(|s| *s)
                .map_err(|e| {
                    XdnaError(format!(
                        "shim symbol {}: {e}",
                        String::from_utf8_lossy(name)
                    ))
                })
        };
        let open_fn: Mm32OpenFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_mm32_open")?) };
        let run_fn: Mm32RunFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_mm32_run")?) };
        let close_fn: Mm32CloseFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_mm32_close")?) };
        let xclbin_c = CString::new(xclbin_path).unwrap();
        let handle = unsafe {
            open_fn(
                xclbin_c.as_ptr(),
                insts.as_ptr(),
                insts.len(),
                m as c_int,
                k as c_int,
                n as c_int,
            )
        };
        if handle.is_null() {
            return Err(XdnaError(
                "rlx_xdna_mm32_open returned null (XRT error on stderr)".into(),
            ));
        }
        Ok(Self {
            handle,
            run_fn,
            close_fn,
            m,
            k,
            n,
            _lib: lib,
        })
    }

    pub fn run(&self, a: &[i32], b: &[i32]) -> Result<Vec<i32>, XdnaError> {
        assert_eq!(a.len(), self.m * self.k, "A must be m*k");
        assert_eq!(b.len(), self.k * self.n, "B must be k*n");
        let mut c = vec![0i32; self.m * self.n];
        let rc = unsafe { (self.run_fn)(self.handle, a.as_ptr(), b.as_ptr(), c.as_mut_ptr()) };
        if rc != 0 {
            return Err(XdnaError(format!("rlx_xdna_mm32_run returned {rc}")));
        }
        Ok(c)
    }
}

impl Drop for NpuMm32 {
    fn drop(&mut self) {
        unsafe { (self.close_fn)(self.handle) }
    }
}

unsafe impl Send for NpuMm32 {}

type Run3OpenFn =
    unsafe extern "C" fn(*const c_char, *const u32, usize, usize, usize, usize) -> *mut c_void;
type Run3RunFn =
    unsafe extern "C" fn(*mut c_void, *const c_void, *const c_void, *mut c_void) -> c_int;
type Run3CloseFn = unsafe extern "C" fn(*mut c_void);

/// Generic 3-buffer **f32** NPU runner — three host buffers of *independent*
/// element counts bound to kernel args 3/4/5, one persistent context. Used by
/// the affine norms (x [rows*cols], gamma+beta [2*cols], out [rows*cols]) and
/// any op needing three differently-sized buffers through one dispatch.
pub struct NpuRun3 {
    handle: *mut c_void,
    run_fn: Run3RunFn,
    close_fn: Run3CloseFn,
    na: usize,
    nb: usize,
    nc: usize,
    _lib: libloading::Library,
}

impl NpuRun3 {
    /// One-time setup. `na`/`nb`/`nc` are f32 element counts for the three buffers.
    pub fn open(
        shim_path: &str,
        xclbin_path: &str,
        insts: &[u32],
        na: usize,
        nb: usize,
        nc: usize,
    ) -> Result<Self, XdnaError> {
        let shim = resolve_shim(shim_path)?;
        let lib = unsafe { libloading::Library::new(&shim) }
            .map_err(|e| XdnaError(format!("dlopen {shim}: {e}")))?;
        let sym = |name: &[u8]| -> Result<*mut c_void, XdnaError> {
            unsafe { lib.get::<*mut c_void>(name) }
                .map(|s| *s)
                .map_err(|e| {
                    XdnaError(format!(
                        "shim symbol {}: {e}",
                        String::from_utf8_lossy(name)
                    ))
                })
        };
        let open_fn: Run3OpenFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_run3_open")?) };
        let run_fn: Run3RunFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_run3_run")?) };
        let close_fn: Run3CloseFn = unsafe { std::mem::transmute(sym(b"rlx_xdna_run3_close")?) };
        let xclbin_c = CString::new(xclbin_path).unwrap();
        let handle = unsafe {
            open_fn(
                xclbin_c.as_ptr(),
                insts.as_ptr(),
                insts.len(),
                na * 4,
                nb * 4,
                nc * 4,
            )
        };
        if handle.is_null() {
            return Err(XdnaError(
                "rlx_xdna_run3_open returned null (XRT error on stderr)".into(),
            ));
        }
        Ok(Self {
            handle,
            run_fn,
            close_fn,
            na,
            nb,
            nc,
            _lib: lib,
        })
    }

    /// Hot path: buffer `a` (na f32) + `b` (nb f32) in, `nc`-f32 out.
    pub fn run(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, XdnaError> {
        assert_eq!(a.len(), self.na, "a must be na");
        assert_eq!(b.len(), self.nb, "b must be nb");
        let mut out = vec![0f32; self.nc];
        let rc = unsafe {
            (self.run_fn)(
                self.handle,
                a.as_ptr() as *const c_void,
                b.as_ptr() as *const c_void,
                out.as_mut_ptr() as *mut c_void,
            )
        };
        if rc != 0 {
            return Err(XdnaError(format!("rlx_xdna_run3_run returned {rc}")));
        }
        Ok(out)
    }
}

impl Drop for NpuRun3 {
    fn drop(&mut self) {
        unsafe { (self.close_fn)(self.handle) }
    }
}

unsafe impl Send for NpuRun3 {}

type PassthroughFn = unsafe extern "C" fn(
    *const c_char, // xclbin
    *const u32,    // insts
    usize,         // ninstr
    c_int,         // n (i32 elems)
    *const i32,    // in
    *mut i32,      // out
) -> c_int;

/// Run an rlx-emitted **DMA-passthrough** overlay on the NPU (`out = in`).
/// Proves an AIE design that rlx generated ([`crate::aie::emit_passthrough`])
/// and compiled ([`crate::compile`]) actually executes on the hardware.
/// `shim_path` empty → `$RLX_XDNA_SHIM`.
pub fn run_passthrough(
    shim_path: &str,
    xclbin_path: &str,
    insts: &[u32],
    input: &[i32],
) -> Result<Vec<i32>, XdnaError> {
    let shim = resolve_shim(shim_path)?;
    let lib = unsafe { libloading::Library::new(&shim) }
        .map_err(|e| XdnaError(format!("dlopen {shim}: {e}")))?;
    let f = unsafe { lib.get::<PassthroughFn>(b"rlx_xdna_run_passthrough") }
        .map_err(|e| XdnaError(format!("shim symbol rlx_xdna_run_passthrough: {e}")))?;
    let xclbin_c = CString::new(xclbin_path).unwrap();
    let mut out = vec![0i32; input.len()];
    let rc = unsafe {
        f(
            xclbin_c.as_ptr(),
            insts.as_ptr(),
            insts.len(),
            input.len() as c_int,
            input.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return Err(XdnaError(format!("rlx_xdna_run_passthrough returned {rc}")));
    }
    Ok(out)
}

/// One-shot INT8 GEMM (open + run + close). Pays cold setup every call — use
/// [`NpuGemm`] directly to amortize it. `shim_path` empty → `$RLX_XDNA_SHIM`.
pub fn run_gemm_i8(
    shim_path: &str,
    xclbin_path: &str,
    insts: &[u32],
    m: usize,
    k: usize,
    n: usize,
    a: &[i8],
    b: &[i8],
) -> Result<Vec<i32>, XdnaError> {
    NpuGemm::open(shim_path, xclbin_path, insts, m, k, n)?.run(a, b)
}
