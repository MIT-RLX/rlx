// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! rlx-xdna — AMD **XDNA / Ryzen AI NPU** backend for RLX (`Device::Xdna`).
//!
//! The XDNA NPU is the AI Engine (`aie2`) tile array on AMD Ryzen AI SoCs, driven
//! on Linux by the in-kernel `amdxdna` driver via `/dev/accel*`. This crate makes
//! it a first-class [`rlx_driver::Device`] that **runs graphs on the NPU** — the
//! forward-inference surface of a transformer and a CNN, plus gradient training —
//! validated bit-exact (or cosine, for the quantized matmul) against the CPU
//! backend on real hardware (a Ryzen Phoenix `npu1` APU, AIE version 1.1).
//!
//! ## What runs on the NPU
//!
//! Graph-level dispatch for `Device::Xdna` lives in `rlx-runtime`'s `XdnaBackend`,
//! which drives the pieces in this crate:
//!
//! - **INT8 GEMM** — the fast matmul path, ~638 GOP/s on Phoenix, via the vendor
//!   `aie::mmul` microkernel overlay ([`npu_gemm`]). The AIE array is an INT8/BF16
//!   MAC engine (no native f32 datapath), so f32 matmuls are per-row/col quantized.
//! - **Transformer** — multi-head causal attention, RoPE (NeoX/GptJ), RMS/Layer/
//!   GroupNorm, softmax, 26 activations, elementwise / reduce / scan / data-movement.
//! - **Vision** — 2-D pooling + im2col (a conv is `im2col → INT8 GEMM` on the NPU).
//! - **Quantization** — Quantize / Dequantize / FakeQuantize (dtype boundary + QAT).
//! - **Training** — backward graphs decompose to these primitives and run on the NPU
//!   (incl. a dynamic-weight GEMM for `xᵀ @ dy`); a host-optimizer SGD loop trains
//!   with the gradient computed on-device.
//!
//! ## The pieces
//!
//! - [`aie`] — the **AIE-MLIR emitter** (pure Rust): rlx emits the per-op AIE kernels
//!   itself, no Python.
//! - [`compile`] — **Python-free overlay compilation** (drives the native `aiecc`
//!   binary → xclbin + instruction stream).
//! - [`npu_gemm`] — the **XRT INT8 GEMM executor** (the fast-matmul path).
//! - [`xrt`] — bindings to the AMD **XRT** userspace runtime + `amdxdna` shim; the
//!   default execution path (`xrt` feature).
//! - [`direct`] — **closest to the metal**: the `amdxdna` DRM-accel ioctl ABI driven
//!   directly on `/dev/accel*`, no XRT / no C++ shim (`direct` feature, Linux-only).
//!   Owns hwctx / BO / exec / syncobj + AXLF-PDI parsing + the TURBO power mode. The
//!   submit + syncobj GEMM path is complete but **parked**: on Phoenix `npu1` (a
//!   kernel-managed-queue part) the firmware won't execute a command XRT runs fine,
//!   and the cause is undiagnosable under Secure Boot lockdown. XRT is the working
//!   path.
//!
//! ## Detection API
//!
//! [`detect`] / [`XdnaStatus`] / [`diagnostic`] report the NPU across a heterogeneous
//! fleet; [`is_available`] is `true` only with a live runtime **and** a configured
//! execution path (an INT8 overlay or the on-demand mlir-aie toolchain). The backend
//! **never silently falls back to the CPU** — a missing runtime is a clear
//! [`XdnaError`], not a masquerade. (The legacy [`SUPPORTED_OPS`] const here is
//! empty; the live op list is `XdnaBackend::supported_ops()` in `rlx-runtime`.)

#[cfg(feature = "xrt")]
pub mod xrt;

/// Native INT8 GEMM executor (drives an MLIR-AIE overlay via the XRT C-API).
#[cfg(feature = "xrt")]
pub mod npu_gemm;

/// **Closest to the metal**: the `amdxdna` DRM-accel ioctl ABI driven directly
/// on `/dev/accel*` — no XRT, no C++ shim. Owns hwctx/BO/exec/syncobj and (later)
/// the user-mode-queue doorbell for zero-syscall dispatch. Linux-only.
#[cfg(feature = "direct")]
pub mod direct;

/// Python-free overlay compilation (drives the native `aiecc` binary).
pub mod compile;

/// rlx emits AIE-MLIR itself (native codegen seam; first increment).
pub mod aie;

/// Legacy detection-seam const, kept empty for back-compat. The **live** op list
/// that `Device::Xdna` lowers is `XdnaBackend::supported_ops()` in `rlx-runtime`
/// (~39 kinds: INT8 GEMM, attention/RoPE/norms, activations, elementwise/reduce/
/// scan, data-movement, quant, pool/im2col, …). This crate provides the emitter +
/// executors that backend drives; it does not run the graph legalizer itself.
pub const SUPPORTED_OPS: &[rlx_ir::OpKind] = &[];

/// A short "north-star" op list describing why the AIE datapath is a good fit
/// (documentation of intent). The AIE-ML tile array is an **INT8 / BF16** MAC
/// engine (Phoenix `npu1` ≈ 16 INT8 TOPS across 4 usable columns); FP32 has no
/// native datapath, so the fast path is quantized matmul plus the conv / attention
/// / norm ops a transformer or vision block needs. The **live** op list actually
/// lowered is `XdnaBackend::supported_ops()` in `rlx-runtime` (~39 kinds); this
/// const is just intent, not a claim.
pub const TARGET_OPS: &[rlx_ir::OpKind] = &[
    rlx_ir::OpKind::DequantMatMul,
    rlx_ir::OpKind::MatMul,
    rlx_ir::OpKind::Conv,
    rlx_ir::OpKind::Attention,
    rlx_ir::OpKind::RmsNorm,
    rlx_ir::OpKind::Softmax,
];

/// Detected state of the XDNA NPU on this host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum XdnaStatus {
    /// No XDNA device found (not a Ryzen AI host, or `amdxdna` not loaded).
    Absent,
    /// NPU hardware + kernel driver present, but no userspace runtime to run on.
    DetectedNoRuntime {
        node: String,
        fw: String,
        product: String,
    },
    /// Hardware **and** a usable XRT/amdxdna userspace runtime are both present:
    /// XRT can talk to the NPU (verified end-to-end with `xrt-smi validate`).
    /// `xrt_lib` is the directory holding the resolved `libxrt_driver_xdna.so*`.
    /// With the mlir-aie toolchain also configured, `Device::Xdna` runs graphs
    /// here (see [`is_available`]).
    RuntimePresent {
        node: String,
        fw: String,
        product: String,
        xrt_lib: String,
    },
}

impl XdnaStatus {
    /// Hardware is physically present (any state but [`Self::Absent`]).
    pub fn hardware_present(&self) -> bool {
        !matches!(self, XdnaStatus::Absent)
    }
}

/// Probe the host for an XDNA NPU. Linux-only; every other OS reports
/// [`XdnaStatus::Absent`] (the `amdxdna` DRM-accel driver is Linux-only).
pub fn detect() -> XdnaStatus {
    #[cfg(target_os = "linux")]
    {
        detect_linux()
    }
    #[cfg(not(target_os = "linux"))]
    {
        XdnaStatus::Absent
    }
}

/// `true` only when rlx can actually **execute a graph** on the NPU: the XRT
/// runtime is present ([`runtime_present`]) AND an INT8 GEMM overlay is
/// configured ([`overlay_from_env`] — shim + xclbin + insts + shape). Gating on
/// a configured overlay keeps this an explicit opt-in and honest: without one
/// there's no kernel to run, so selection won't dispatch here (no masquerade).
pub fn is_available() -> bool {
    runtime_present() && (overlay_from_env().is_some() || op_compile_available())
}

/// `true` when the native mlir-aie toolchain is configured (`AIECC` + `PEANO`
/// both set and present), so the backend can compile elementwise / softmax
/// kernels **on demand** — no precompiled matmul overlay needed for those ops.
/// The other half of [`is_available`]'s opt-in (overlay for MatMul, toolchain
/// for the compile-on-demand op path).
pub fn op_compile_available() -> bool {
    let ok = |k: &str| {
        std::env::var(k).map(|v| std::path::Path::new(&v).exists()).unwrap_or(false)
    };
    ok("AIECC") && ok("PEANO")
}

/// A configured INT8 GEMM overlay: the compiled MLIR-AIE artifacts + the C++
/// shim + the shape they were built for. Read from the environment:
///   `RLX_XDNA_SHIM`   — `librlx_xdna_shim.so`
///   `RLX_XDNA_XCLBIN` — the overlay `.xclbin`
///   `RLX_XDNA_INSTS`  — the paired `insts_*.bin`
///   `RLX_XDNA_GEMM`   — `"M,K,N"` the overlay was compiled for
#[derive(Debug, Clone)]
pub struct Overlay {
    pub shim: String,
    pub xclbin: String,
    pub insts: String,
    pub m: usize,
    pub k: usize,
    pub n: usize,
}

/// Read the overlay config from the environment, or `None` if not fully set /
/// any file is missing.
pub fn overlay_from_env() -> Option<Overlay> {
    let path = |k: &str| -> Option<String> {
        let v = std::env::var(k).ok()?;
        std::path::Path::new(&v).exists().then_some(v)
    };
    let shim = path("RLX_XDNA_SHIM")?;
    let xclbin = path("RLX_XDNA_XCLBIN")?;
    let insts = path("RLX_XDNA_INSTS")?;
    let mut mkn = std::env::var("RLX_XDNA_GEMM")
        .ok()?
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect::<Vec<_>>()
        .into_iter();
    let (m, k, n) = (mkn.next()?, mkn.next()?, mkn.next()?);
    Some(Overlay {
        shim,
        xclbin,
        insts,
        m,
        k,
        n,
    })
}

/// `true` when the **XRT userspace runtime** can talk to the NPU (hardware +
/// `libxrt_driver_xdna.so` resolvable) — i.e. `xrt-smi`/`pyxrt` work. This is a
/// lower bar than [`is_available`], which additionally requires a configured
/// execution path (an INT8 overlay or the on-demand mlir-aie toolchain) before it
/// will dispatch a graph.
pub fn runtime_present() -> bool {
    matches!(detect(), XdnaStatus::RuntimePresent { .. })
}

/// `true` if the NPU hardware is physically present, regardless of runtime.
/// Use for inventory; use [`is_available`] to gate graph execution.
pub fn hardware_present() -> bool {
    detect().hardware_present()
}

/// One-line human diagnostic describing what was found and what's missing.
pub fn diagnostic() -> String {
    match detect() {
        XdnaStatus::Absent => "AMD XDNA NPU: not found (not a Ryzen AI host, or the \
             `amdxdna` driver is not loaded)."
            .to_string(),
        XdnaStatus::DetectedNoRuntime { node, fw, product } => format!(
            "AMD XDNA NPU detected at {node} ({product}, fw {fw}) but no userspace runtime \
             is available. rlx-xdna needs AMD XRT + the amdxdna shim (libxrt_coreutil.so + \
             libxrt_driver_xdna.so; set XILINX_XRT) and an AIE-compiled kernel image to \
             execute. Refusing to run on the NPU — no CPU fallback."
        ),
        XdnaStatus::RuntimePresent { node, fw, product, xrt_lib } => format!(
            "AMD XDNA NPU live at {node} ({product}, fw {fw}); XRT runtime present ({xrt_lib}). \
             `Device::Xdna` runs graphs here once the mlir-aie toolchain is configured (set AIECC \
             + PEANO for the compile-on-demand op path, or RLX_XDNA_GEMM for a precompiled INT8 \
             overlay) — the AIE-ML tiles are an INT8/BF16 engine, so f32 matmuls run quantized. \
             No CPU fallback."
        ),
    }
}

/// Error type for the detection seam / diagnostics (`ensure_executable`,
/// `diagnostic`). Execution errors on the live path surface through the
/// `rlx-runtime` `XdnaBackend` instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct XdnaError(pub String);

impl std::fmt::Display for XdnaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for XdnaError {}

/// Legacy detection-seam probe used for **fleet inventory / diagnostics**, not the
/// live execution path — the real graph runner is `rlx-runtime`'s `XdnaBackend`
/// (which drives [`npu_gemm`] / [`aie`] / [`compile`] directly). Returns a
/// [`XdnaError`] whose message ([`diagnostic`]) pinpoints what's missing on this
/// host (no NPU / no runtime / runtime-but-no-overlay), so inventory tooling can
/// report an honest "detected, not runnable" instead of a CPU masquerade.
pub fn ensure_executable() -> Result<(), XdnaError> {
    Err(XdnaError(diagnostic()))
}

#[cfg(target_os = "linux")]
fn detect_linux() -> XdnaStatus {
    use std::path::Path;

    let base = Path::new("/sys/class/accel");
    let Ok(entries) = std::fs::read_dir(base) else {
        return XdnaStatus::Absent;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned(); // e.g. "accel0"
        let devdir = base.join(&name).join("device");
        let product = read_trim(&devdir.join("vbnv")).unwrap_or_default();
        let fw = read_trim(&devdir.join("fw_version")).unwrap_or_default();

        // An amdxdna-backed NPU: the bound driver is `amdxdna`, or the product
        // string names a Ryzen AI NPU (vbnv like "RyzenAI-npu1").
        let plc = product.to_ascii_lowercase();
        let is_xdna =
            driver_is_amdxdna(&devdir) || plc.contains("ryzenai") || plc.contains("npu");
        if !is_xdna {
            continue;
        }
        // Hardware is present as soon as sysfs enumerates the amdxdna device.
        // The char device `/dev/accelN` can flap (runtime-PM autosuspend removes
        // it when the NPU is idle, re-adds it on use), so it does NOT gate
        // presence — otherwise a suspended-but-installed NPU would read Absent.
        // XRT resumes the device on open, so the runtime doesn't need it up now.
        let node = format!("/dev/{name}");
        return match xrt_shim_dir() {
            Some(xrt_lib) => XdnaStatus::RuntimePresent {
                node,
                fw,
                product,
                xrt_lib,
            },
            None => XdnaStatus::DetectedNoRuntime { node, fw, product },
        };
    }
    XdnaStatus::Absent
}

#[cfg(target_os = "linux")]
fn driver_is_amdxdna(devdir: &std::path::Path) -> bool {
    std::fs::read_link(devdir.join("driver"))
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .is_some_and(|drv| drv == "amdxdna")
}

/// Directory holding a usable XRT runtime — one that has BOTH the core
/// (`libxrt_coreutil.so*`) and the amdxdna shim (`libxrt_driver_xdna.so*`). Any
/// soname (`.so`, `.so.2`, `.so.2.21.75`) counts, since the Ubuntu-packaged XRT
/// ships only versioned sonames. Search order: `$RLX_XDNA_XRT_LIB`, each
/// `$LD_LIBRARY_PATH` entry (how the extracted-in-$HOME runtime is reached),
/// `$XILINX_XRT/lib`, `/opt/xilinx/xrt/lib`, and the system multiarch libdir.
#[cfg(target_os = "linux")]
fn xrt_shim_dir() -> Option<String> {
    let mut dirs: Vec<String> = Vec::new();
    if let Ok(d) = std::env::var("RLX_XDNA_XRT_LIB") {
        dirs.push(d);
    }
    if let Ok(p) = std::env::var("LD_LIBRARY_PATH") {
        dirs.extend(p.split(':').filter(|s| !s.is_empty()).map(String::from));
    }
    if let Ok(root) = std::env::var("XILINX_XRT") {
        dirs.push(format!("{root}/lib"));
    }
    dirs.push("/opt/xilinx/xrt/lib".to_string());
    dirs.push("/usr/lib/x86_64-linux-gnu".to_string());

    dirs.into_iter()
        .find(|d| dir_has_lib(d, "libxrt_coreutil.so") && dir_has_lib(d, "libxrt_driver_xdna.so"))
}

/// Does `dir` contain a file whose name starts with `stem` (matches any soname
/// version suffix)?
#[cfg(target_os = "linux")]
fn dir_has_lib(dir: &str, stem: &str) -> bool {
    std::fs::read_dir(dir).ok().is_some_and(|mut rd| {
        rd.any(|e| {
            e.ok()
                .map(|e| e.file_name().to_string_lossy().starts_with(stem))
                .unwrap_or(false)
        })
    })
}

#[cfg(target_os = "linux")]
fn read_trim(p: &std::path::Path) -> Option<String> {
    std::fs::read_to_string(p)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics_and_is_consistent() {
        let s = detect();
        // is_available = runtime present AND (an overlay OR the on-demand toolchain).
        assert_eq!(
            is_available(),
            runtime_present() && (overlay_from_env().is_some() || op_compile_available())
        );
        assert_eq!(
            runtime_present(),
            matches!(s, XdnaStatus::RuntimePresent { .. })
        );
        assert_eq!(hardware_present(), s.hardware_present());
    }

    #[test]
    fn diagnostic_seam_names_device_no_fallback() {
        // The diagnostics probe always yields an actionable, device-named message
        // (no CPU masquerade), whatever the host state. Live graph execution goes
        // through `rlx-runtime`'s `XdnaBackend`, not this seam.
        let err = ensure_executable().expect_err("ensure_executable is a diagnostics probe");
        assert!(!err.0.is_empty());
        assert!(
            err.0.contains("XDNA") || err.0.contains("NPU"),
            "diagnostic should name the device: {}",
            err.0
        );
    }

    #[test]
    fn legacy_supported_ops_const_is_empty() {
        // Back-compat: the live op list is `XdnaBackend::supported_ops()` in
        // `rlx-runtime`; this legacy const stays empty.
        assert!(SUPPORTED_OPS.is_empty());
    }
}
