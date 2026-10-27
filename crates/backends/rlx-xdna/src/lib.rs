// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! rlx-xdna — AMD **XDNA / Ryzen AI NPU** backend for RLX (`Device::Xdna`).
//!
//! The XDNA NPU is the AI Engine (`aie2`) tile array on AMD Ryzen AI SoCs, driven
//! on Linux by the in-kernel `amdxdna` driver via `/dev/accel*`. This crate makes
//! it a first-class `rlx_driver::Device` that **runs graphs on the NPU** — the
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
//!   `aie::mmul` microkernel overlay (`npu_gemm`). The AIE array is an INT8/BF16
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
//! - `npu_gemm` — the **XRT INT8 GEMM executor** (the fast-matmul path).
//! - `xrt` — bindings to the AMD **XRT** userspace runtime + `amdxdna` shim; the
//!   default execution path (`xrt` feature).
//! - `direct` — **closest to the metal**: the `amdxdna` DRM-accel ioctl ABI driven
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
pub mod setup;
pub mod toolchain;

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

/// The rlx XRT shim (`librlx_xdna_shim.so`), or `None` when [`shim_path`]
/// cannot resolve or build one.
///
/// This is rlx's own `extern "C"` wrapper built from `csrc/xrt_gemm_shim.cpp`,
/// **not** XRT's `libxrt_driver_xdna.so`. Pointing it at the latter loads fine
/// and then fails at first use with `undefined symbol: rlx_xdna_io_open`. Build:
///
/// ```sh
/// g++ -O2 -fPIC -shared -std=c++17 -I$XILINX_XRT/include \
///     -o librlx_xdna_shim.so csrc/xrt_gemm_shim.cpp \
///     -L$XILINX_XRT/lib -Wl,-rpath,$XILINX_XRT/lib -lxrt_coreutil
/// ```
///
/// When `RLX_XDNA_SHIM` is unset, rlx builds that command itself — see
/// [`shim_path`].
pub fn shim_from_env() -> Option<String> {
    shim_path().ok()
}

/// rlx's XRT shim: `$RLX_XDNA_SHIM` when set, otherwise built once from the
/// bundled `csrc/xrt_gemm_shim.cpp` with the host C++ compiler against the XRT
/// runtime [`detect`] found, and cached under `$XDG_CACHE_HOME/rlx-xdna`
/// (issue #11). `Err` says why neither worked. `RLX_XDNA_NO_SHIM_BUILD=1`
/// disables the build.
pub fn shim_path() -> Result<String, String> {
    match rlx_ir::env::read("RLX_XDNA_SHIM") {
        Ok(v) if std::path::Path::new(&v).exists() => Ok(v),
        Ok(v) => Err(format!("RLX_XDNA_SHIM={v} does not exist")),
        Err(_) => {
            use std::sync::OnceLock;
            static BUILT: OnceLock<Result<String, String>> = OnceLock::new();
            BUILT.get_or_init(build_shim).clone()
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn build_shim() -> Result<String, String> {
    Err("RLX_XDNA_SHIM is not set".into())
}

#[cfg(target_os = "linux")]
fn build_shim() -> Result<String, String> {
    use std::hash::{Hash, Hasher};
    use std::path::{Path, PathBuf};
    const SRC: &str = include_str!("../csrc/xrt_gemm_shim.cpp");
    let unset = "RLX_XDNA_SHIM is not set";
    if rlx_ir::env::flag("RLX_XDNA_NO_SHIM_BUILD") {
        return Err(format!(
            "{unset} (auto-build disabled by RLX_XDNA_NO_SHIM_BUILD)"
        ));
    }
    let lib_dir = xrt_shim_dir()
        .ok_or_else(|| format!("{unset}, and no XRT runtime was found to build it against"))?;
    // Headers come from the -dev package (`libxrt-dev` on Debian/Ubuntu) or an
    // XRT install tree; the runtime packages alone do not ship them.
    let include = std::env::var("XILINX_XRT")
        .ok()
        .map(|r| format!("{r}/include"))
        .into_iter()
        .chain([
            "/usr/include".to_string(),
            "/opt/xilinx/xrt/include".to_string(),
        ])
        .find(|d| Path::new(d).join("xrt/xrt_device.h").exists())
        .ok_or_else(|| {
            format!(
                "{unset}, and building it needs the XRT headers (xrt/xrt_device.h) — install \
                 libxrt-dev, or set XILINX_XRT to an XRT tree with include/"
            )
        })?;
    // Distro runtime packages ship only versioned sonames, which `-l` cannot
    // name; link the file directly then.
    let coreutil: Vec<String> = if Path::new(&lib_dir).join("libxrt_coreutil.so").exists() {
        vec![format!("-L{lib_dir}"), "-lxrt_coreutil".into()]
    } else {
        let versioned = std::fs::read_dir(&lib_dir)
            .ok()
            .and_then(|rd| {
                rd.flatten().map(|e| e.path()).find(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("libxrt_coreutil.so"))
                })
            })
            .ok_or_else(|| format!("{unset}, and libxrt_coreutil.so* vanished from {lib_dir}"))?;
        vec![versioned.to_string_lossy().into_owned()]
    };
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".into());
    let mut args: Vec<String> = vec![
        "-O2".into(),
        "-fPIC".into(),
        "-shared".into(),
        "-std=c++17".into(),
        format!("-I{include}"),
    ];
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (SRC, &cxx, &args, &coreutil, &lib_dir).hash(&mut h);
    let key = h.finish();

    let cache = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("rlx-xdna");
    let out = cache.join(format!("librlx_xdna_shim-{key:016x}.so"));
    if out.exists() {
        return Ok(out.to_string_lossy().into_owned());
    }
    std::fs::create_dir_all(&cache)
        .map_err(|e| format!("{unset}; cannot create {}: {e}", cache.display()))?;
    let src = cache.join(format!("xrt_gemm_shim-{key:016x}.cpp"));
    std::fs::write(&src, SRC)
        .map_err(|e| format!("{unset}; cannot write {}: {e}", src.display()))?;
    // Build to a private name, then rename: a concurrent process must never
    // dlopen a half-written library.
    let tmp = cache.join(format!(
        "librlx_xdna_shim-{key:016x}.so.{}",
        std::process::id()
    ));
    args.push("-o".into());
    args.push(tmp.to_string_lossy().into_owned());
    args.push(src.to_string_lossy().into_owned());
    args.extend(coreutil);
    args.push(format!("-Wl,-rpath,{lib_dir}"));
    eprintln!(
        "[rlx-xdna] RLX_XDNA_SHIM unset — building {}",
        out.display()
    );
    let output = std::process::Command::new(&cxx)
        .args(&args)
        .output()
        .map_err(|e| format!("{unset}, and running `{cxx}` to build it failed: {e} (set CXX)"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(12).collect();
        let tail: Vec<&str> = tail.into_iter().rev().collect();
        return Err(format!(
            "{unset}, and building it failed: `{cxx} {}`:\n{}",
            args.join(" "),
            tail.join("\n")
        ));
    }
    std::fs::rename(&tmp, &out)
        .map_err(|e| format!("{unset}; cannot install {}: {e}", out.display()))?;
    Ok(out.to_string_lossy().into_owned())
}

/// `true` only when rlx can actually **execute a graph** on the NPU: the XRT
/// runtime is present ([`runtime_present`]), the rlx shim is resolvable
/// ([`shim_from_env`]), AND there is a kernel path — a precompiled INT8 GEMM
/// overlay ([`overlay_from_env`]) or the on-demand mlir-aie toolchain
/// ([`op_compile_available`]).
///
/// The shim is checked here, not just inside [`overlay_from_env`], because
/// **both** kernel paths dlopen it. Without that check `AIECC`+`PEANO` alone
/// reported the device as available and then the backend panicked at first
/// compile (`no shim path`) — the one failure mode this backend's "no CPU
/// masquerade" contract is supposed to rule out. `Backend::compile` returns a
/// graph, not a `Result`, so being honest here is the only way to fail cleanly.
pub fn is_available() -> bool {
    runtime_present()
        && host_blockers().is_empty()
        && shim_from_env().is_some()
        && (overlay_from_env().is_some() || op_compile_available())
}

/// Host conditions that stop XRT from driving a detected NPU however rlx is
/// configured, each phrased as the fix. Empty when none were found.
///
/// * An XDNA 2 (Strix-class) part: the [`aie`] emitters and overlays target
///   Phoenix / Hawk Point `npu1` only, and an `npu1` xclbin cannot load on AIE2P.
/// * A device node the current user cannot open read/write (XRT opens it).
pub fn host_blockers() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        let Some(name) = xdna_accel_name() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let devdir = std::path::Path::new("/sys/class/accel")
            .join(&name)
            .join("device");
        let pci_id = read_trim(&devdir.join("device"))
            .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok());
        // The PCI id table names known parts; the live column count catches
        // XDNA 2 ids not in it yet.
        let geometry = aie_geometry();
        if pci_id.is_some_and(|id| XDNA2_PCI_IDS.contains(&id))
            || geometry.is_some_and(|g| g.is_xdna2())
        {
            let what = match (pci_id, geometry) {
                (_, Some(g)) if g.is_xdna2() => format!("{} AIE columns", g.cols),
                (Some(id), _) => format!("PCI id {id:#06x}"),
                _ => String::new(),
            };
            out.push(format!(
                "this is an XDNA 2 (Strix-class, AIE2P) NPU ({what}); rlx-xdna's kernels \
                 target XDNA 1 (Phoenix / Hawk Point `npu1`) only"
            ));
        }
        // The node can be absent while the NPU is runtime-suspended (see
        // `detect_linux`); only an existing node with the wrong mode is a blocker.
        let node = format!("/dev/accel/{name}");
        if let Err(e) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&node)
            && e.kind() == std::io::ErrorKind::PermissionDenied
        {
            out.push(format!(
                "{node} is not read/write for this user — add the user to the `render` \
                 group (or fix the udev rule), then log in again"
            ));
        }
        out
    }
    #[cfg(not(target_os = "linux"))]
    {
        Vec::new()
    }
}

/// Live AIE tile-array geometry, read from the driver with the metadata-only
/// `DRM_IOCTL_AMDXDNA_GET_INFO` queries (no DMA buffers, no root, no memlock).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AieGeometry {
    /// Columns the driver reports. Phoenix / Hawk Point report 5, of which the
    /// MLIR-AIE `npu1` device uses 4 for compute; XDNA 2 parts report 8.
    pub cols: u16,
    pub rows: u16,
    pub col_size: u32,
    /// AIE array `(major, minor)` version.
    pub version: (u32, u32),
}

impl AieGeometry {
    /// An XDNA 2 (AIE2P) array: 8 columns or more.
    pub fn is_xdna2(&self) -> bool {
        self.cols >= 8
    }
}

/// Query the NPU's [`AieGeometry`] from the driver. `None` without the
/// `direct` feature, off Linux, or when the device node cannot be opened
/// (absent while runtime-suspended, or no permission — see [`host_blockers`]).
pub fn aie_geometry() -> Option<AieGeometry> {
    #[cfg(all(feature = "direct", target_os = "linux"))]
    {
        let npu = direct::Npu::open(&format!("/dev/accel/{}", xdna_accel_name()?)).ok()?;
        let (cols, rows, col_size) = npu.aie_metadata().ok()?;
        Some(AieGeometry {
            cols,
            rows,
            col_size,
            version: npu.aie_version().ok()?,
        })
    }
    #[cfg(not(all(feature = "direct", target_os = "linux")))]
    {
        None
    }
}

/// PCI device ids of XDNA 2 NPUs (Strix / Strix Halo / Krackan share one).
/// Phoenix and Hawk Point (`npu1`, the parts rlx-xdna targets) are `0x1502`.
#[cfg(target_os = "linux")]
const XDNA2_PCI_IDS: &[u32] = &[0x17f0];

/// Soft `Max locked memory` from `/proc/<pid>/limits` text, in bytes;
/// `None` when unlimited or unreadable.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn memlock_soft_bytes(limits: &str) -> Option<u64> {
    let line = limits
        .lines()
        .find(|l| l.starts_with("Max locked memory"))?;
    line.split_whitespace().nth(3)?.parse().ok()
}

/// A low `RLIMIT_MEMLOCK` makes XRT's buffer-object mmap fail with `EAGAIN`
/// once buffers outgrow it (distro default is 8 MiB). A warning, not a
/// blocker: small graphs fit, so this cannot decide availability.
fn memlock_warning() -> Option<String> {
    #[cfg(target_os = "linux")]
    {
        let bytes = memlock_soft_bytes(&std::fs::read_to_string("/proc/self/limits").ok()?)?;
        (bytes < 256 << 20).then(|| {
            format!(
                "RLIMIT_MEMLOCK is {} MiB; XRT buffer mmaps fail with EAGAIN past it — raise \
                 it (e.g. `@render - memlock unlimited` in /etc/security/limits.d/) and log in again",
                bytes >> 20
            )
        })
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// `true` when the native mlir-aie toolchain is available — `AIECC` + `PEANO`
/// set, or found by [`toolchain::toolchain`] — so the backend can compile elementwise / softmax
/// kernels **on demand** — no precompiled matmul overlay needed for those ops.
/// The other half of [`is_available`]'s opt-in (overlay for MatMul, toolchain
/// for the compile-on-demand op path).
pub fn op_compile_available() -> bool {
    toolchain::toolchain().is_ok() && toolchain_probe().is_ok()
}

/// Why env var `key` does not name an existing path (`"KEY is not set"` /
/// `"KEY=… does not exist"`), or `None` when it does. Detection used to answer
/// a bare `None` here and the user saw only "no such device/driver".
fn env_path_problem(key: &str) -> Option<String> {
    match rlx_ir::env::read(key) {
        Err(_) => Some(format!("{key} is not set")),
        Ok(v) if !std::path::Path::new(&v).exists() => Some(format!("{key}={v} does not exist")),
        Ok(_) => None,
    }
}

/// Whether the configured toolchain can actually compile what **this** version
/// of rlx emits — not merely whether `aiecc` exists on disk.
///
/// The path-existence check above is not enough, and that gap was live: rlx
/// emitted `aie.logical_tile`, no released mlir-aie knew that op, and XDNA
/// still advertised itself as dispatchable while every kernel failed to
/// compile. `is_available` feeds device SELECTION and there is no CPU fallback
/// behind it, so a wrong answer here is a hang or a hard failure, not a slow
/// path.
///
/// Compiles a minimal real design once per process (~0.4 s) and caches the
/// verdict; `Err` carries why (the compiler's own error), so [`diagnostic`]
/// can show it. `RLX_XDNA_SKIP_PROBE=1` trusts the paths instead, for a host
/// where spawning the compiler during device selection is unwanted.
fn toolchain_probe() -> Result<(), String> {
    use std::sync::OnceLock;
    static VERDICT: OnceLock<Result<(), String>> = OnceLock::new();
    VERDICT
        .get_or_init(|| {
            if rlx_ir::env::flag("RLX_XDNA_SKIP_PROBE") {
                return Ok(());
            }
            let tc = toolchain::toolchain()?;
            let (aiecc, peano) = (tc.aiecc_str(), tc.peano_str());
            let dir = std::env::temp_dir().join("rlx_xdna_probe");
            // Clear first. `compile_overlay` reports success by checking its outputs
            // exist, so a leftover `probe.xclbin` from an earlier good run makes a
            // BROKEN toolchain probe clean — verified: pointing AIECC at /bin/echo
            // still reported XDNA dispatchable until this line existed.
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir)
                .map_err(|e| format!("cannot create probe dir {}: {e}", dir.display()))?;
            let mlir = dir.join("probe.mlir");
            // The same shapes the emitters use, so a dialect change that breaks
            // them breaks the probe too.
            std::fs::write(&mlir, crate::aie::emit_passthrough(1024, 1024))
                .map_err(|e| format!("cannot write {}: {e}", mlir.display()))?;
            let xclbin = dir.join("probe.xclbin");
            let insts = dir.join("probe_insts.bin");
            // aiecc's scratch dir is wiped per attempt, so it must not be `dir`
            // itself — that deleted probe.mlir before aiecc read it (issue #9).
            let build = dir.join("build");
            let utf8 = |p: &std::path::Path| {
                p.to_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("probe path is not UTF-8: {}", p.display()))
            };
            let (mlir, build, xclbin, insts) =
                (utf8(&mlir)?, utf8(&build)?, utf8(&xclbin)?, utf8(&insts)?);
            let spec = crate::compile::OverlaySpec {
                aiecc: &aiecc,
                peano: &peano,
                mlir: &mlir,
                tmpdir: &build,
                out_xclbin: &xclbin,
                out_insts: &insts,
            };
            crate::compile::compile_overlay(&spec).map(|_| ()).map_err(|e| {
            eprintln!(
                "[rlx-xdna] toolchain probe FAILED — AIECC/PEANO are set but cannot compile \
                 rlx's AIE-MLIR. Treating XDNA as unavailable rather than dispatching to it.\n{e}"
            );
            // aiecc output can run to pages; the tail holds the actual error.
            let msg = e.to_string();
            let tail: Vec<&str> = msg.lines().rev().take(12).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            format!(
                "toolchain probe failed to compile rlx's AIE-MLIR with AIECC={aiecc} \
                 (set RLX_XDNA_SKIP_PROBE=1 to bypass): {}",
                tail.join("\n")
            )
        })
        })
        .clone()
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
/// any file is missing. [`overlay_problems`] says which.
pub fn overlay_from_env() -> Option<Overlay> {
    overlay_checked().ok()
}

/// Every reason [`overlay_from_env`] is `None` (empty when it is `Some`).
pub fn overlay_problems() -> Vec<String> {
    overlay_checked().err().unwrap_or_default()
}

fn overlay_checked() -> Result<Overlay, Vec<String>> {
    let mut problems: Vec<String> = ["RLX_XDNA_XCLBIN", "RLX_XDNA_INSTS"]
        .into_iter()
        .filter_map(env_path_problem)
        .collect();
    let shim = shim_path().map_err(|e| problems.push(e)).ok();
    let mkn: Option<Vec<usize>> = match rlx_ir::env::var("RLX_XDNA_GEMM") {
        None => {
            problems.push("RLX_XDNA_GEMM is not set".into());
            None
        }
        Some(v) => {
            let dims: Result<Vec<usize>, _> =
                v.split(',').map(|s| s.trim().parse::<usize>()).collect();
            match dims {
                Ok(d) if d.len() == 3 => Some(d),
                _ => {
                    problems.push(format!("RLX_XDNA_GEMM={v} is not \"M,K,N\""));
                    None
                }
            }
        }
    };
    match mkn {
        Some(d) if problems.is_empty() => {
            let read = |k: &str| rlx_ir::env::read(k).unwrap_or_default();
            Ok(Overlay {
                shim: shim.unwrap_or_default(),
                xclbin: read("RLX_XDNA_XCLBIN"),
                insts: read("RLX_XDNA_INSTS"),
                m: d[0],
                k: d[1],
                n: d[2],
            })
        }
        _ => Err(problems),
    }
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
        XdnaStatus::Absent => format!(
            "AMD XDNA NPU: not found (not a Ryzen AI host, or the `amdxdna` driver is not \
             loaded). {}",
            absent_reason()
        ),
        XdnaStatus::DetectedNoRuntime { node, fw, product } => format!(
            "AMD XDNA NPU detected at {node} ({product}, fw {fw}) but no userspace runtime \
             is available. rlx-xdna needs AMD XRT + the amdxdna shim (libxrt_coreutil.so* + \
             libxrt_driver_xdna.so*, both in one directory) and an AIE-compiled kernel image \
             to execute. Searched: {}. Set RLX_XDNA_XRT_LIB or XILINX_XRT to the directory \
             holding them. Refusing to run on the NPU — no CPU fallback.",
            xrt_search_dirs().join(", ")
        ),
        XdnaStatus::RuntimePresent {
            node,
            fw,
            product,
            xrt_lib,
        } => format!(
            "AMD XDNA NPU live at {node} ({product}, fw {fw}{geometry}); XRT runtime present \
             ({xrt_lib}). Still needed to execute: {missing}. The AIE-ML tiles are an INT8/BF16 \
             engine, so f32 matmuls run quantized. No CPU fallback.",
            geometry = aie_geometry()
                .map(|g| format!(
                    ", AIE v{}.{} {}x{} array",
                    g.version.0, g.version.1, g.cols, g.rows
                ))
                .unwrap_or_default(),
            missing = {
                let mut m: Vec<String> = host_blockers();
                if let Err(why) = shim_path() {
                    m.push(format!(
                        "RLX_XDNA_SHIM (rlx's own librlx_xdna_shim.so, built from \
                         csrc/xrt_gemm_shim.cpp — NOT libxrt_driver_xdna.so): {why}"
                    ));
                }
                if overlay_from_env().is_none() && !op_compile_available() {
                    let toolchain = match toolchain::toolchain() {
                        Err(e) => e,
                        Ok(_) => toolchain_probe().err().unwrap_or_default(),
                    };
                    m.push(format!(
                        "a kernel path — either AIECC (native mlir_aie/bin/aiecc) + PEANO \
                         (llvm-aie install dir) for compile-on-demand [{toolchain}], or \
                         RLX_XDNA_SHIM/XCLBIN/INSTS/GEMM for a precompiled INT8 overlay [{}]",
                        overlay_problems().join(", ")
                    ));
                }
                let mut out = if m.is_empty() {
                    "nothing — `Device::Xdna` is ready".to_string()
                } else {
                    m.join("; ")
                };
                if let Some(w) = memlock_warning() {
                    out.push_str(&format!(" (warning: {w})"));
                }
                out
            }
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
/// (which drives `npu_gemm` / [`aie`] / [`compile`] directly). Returns a
/// [`XdnaError`] whose message ([`diagnostic`]) pinpoints what's missing on this
/// host (no NPU / no runtime / runtime-but-no-overlay), so inventory tooling can
/// report an honest "detected, not runnable" instead of a CPU masquerade.
pub fn ensure_executable() -> Result<(), XdnaError> {
    Err(XdnaError(diagnostic()))
}

/// What detection looked at when it reported [`XdnaStatus::Absent`].
fn absent_reason() -> String {
    #[cfg(target_os = "linux")]
    {
        let base = std::path::Path::new("/sys/class/accel");
        let Ok(entries) = std::fs::read_dir(base) else {
            return "/sys/class/accel does not exist (no DRM-accel driver loaded; \
                    `modprobe amdxdna` needs Linux ≥ 6.14 or the out-of-tree module)."
                .into();
        };
        let seen: Vec<String> = entries
            .flatten()
            .map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                let devdir = base.join(&name).join("device");
                let driver = std::fs::read_link(devdir.join("driver"))
                    .ok()
                    .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                    .unwrap_or_else(|| "?".into());
                let product = read_trim(&devdir.join("vbnv")).unwrap_or_else(|| "?".into());
                format!("{name} (driver {driver}, vbnv {product})")
            })
            .collect();
        if seen.is_empty() {
            "/sys/class/accel is empty.".into()
        } else {
            format!("No amdxdna device among: {}.", seen.join(", "))
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        "The amdxdna driver is Linux-only.".into()
    }
}

/// The `/sys/class/accel` entry (e.g. `accel0`) of the first amdxdna NPU.
#[cfg(target_os = "linux")]
fn xdna_accel_name() -> Option<String> {
    let base = std::path::Path::new("/sys/class/accel");
    std::fs::read_dir(base).ok()?.flatten().find_map(|entry| {
        let name = entry.file_name().to_string_lossy().into_owned();
        let devdir = base.join(&name).join("device");
        // An amdxdna-backed NPU: the bound driver is `amdxdna`, or the product
        // string names a Ryzen AI NPU (vbnv like "RyzenAI-npu1").
        let plc = read_trim(&devdir.join("vbnv"))
            .unwrap_or_default()
            .to_ascii_lowercase();
        (driver_is_amdxdna(&devdir) || plc.contains("ryzenai") || plc.contains("npu"))
            .then_some(name)
    })
}

#[cfg(target_os = "linux")]
fn detect_linux() -> XdnaStatus {
    let Some(name) = xdna_accel_name() else {
        return XdnaStatus::Absent;
    };
    let devdir = std::path::Path::new("/sys/class/accel")
        .join(&name)
        .join("device");
    let product = read_trim(&devdir.join("vbnv")).unwrap_or_default();
    let fw = read_trim(&devdir.join("fw_version")).unwrap_or_default();
    // Hardware is present as soon as sysfs enumerates the amdxdna device.
    // The char device `/dev/accel/accelN` can flap (runtime-PM autosuspend
    // removes it when the NPU is idle, re-adds it on use), so it does NOT gate
    // presence — otherwise a suspended-but-installed NPU would read Absent.
    // XRT resumes the device on open, so the runtime doesn't need it up now.
    let node = format!("/dev/accel/{name}");
    match xrt_shim_dir() {
        Some(xrt_lib) => XdnaStatus::RuntimePresent {
            node,
            fw,
            product,
            xrt_lib,
        },
        None => XdnaStatus::DetectedNoRuntime { node, fw, product },
    }
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
/// `$XILINX_XRT/lib`, `/opt/xilinx/xrt/lib`, then the distro libdirs.
#[cfg(target_os = "linux")]
fn xrt_shim_dir() -> Option<String> {
    xrt_search_dirs()
        .into_iter()
        .find(|d| dir_has_lib(d, "libxrt_coreutil.so") && dir_has_lib(d, "libxrt_driver_xdna.so"))
}

/// The directories [`xrt_shim_dir`] searches, in order.
fn xrt_search_dirs() -> Vec<String> {
    let mut dirs: Vec<String> = Vec::new();
    if let Some(d) = rlx_ir::env::var("RLX_XDNA_XRT_LIB") {
        dirs.push(d);
    }
    if let Ok(p) = std::env::var("LD_LIBRARY_PATH") {
        dirs.extend(p.split(':').filter(|s| !s.is_empty()).map(String::from));
    }
    if let Ok(root) = std::env::var("XILINX_XRT") {
        dirs.push(format!("{root}/lib"));
    }
    dirs.push("/opt/xilinx/xrt/lib".to_string());
    // Debian/Ubuntu multiarch, Arch (`/usr/lib`), Fedora (`/usr/lib64`).
    dirs.push("/usr/lib/x86_64-linux-gnu".to_string());
    dirs.push("/usr/lib".to_string());
    dirs.push("/usr/lib64".to_string());
    dirs
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
            runtime_present()
                && host_blockers().is_empty()
                && shim_from_env().is_some()
                && (overlay_from_env().is_some() || op_compile_available())
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
    fn overlay_problems_name_each_missing_var() {
        let missing = std::env::temp_dir().join("rlx_xdna_no_such_file.xclbin");
        let present = std::env::current_exe().unwrap();
        let _shim = rlx_ir::env::scoped("RLX_XDNA_SHIM", present.to_str().unwrap());
        let _xclbin = rlx_ir::env::scoped("RLX_XDNA_XCLBIN", missing.to_str().unwrap());
        let _insts = rlx_ir::env::scoped_remove("RLX_XDNA_INSTS");
        let _gemm = rlx_ir::env::scoped("RLX_XDNA_GEMM", "64,x,64");

        let p = overlay_problems();
        assert!(overlay_from_env().is_none());
        assert_eq!(p.len(), 3, "{p:?}");
        assert!(p[0].starts_with("RLX_XDNA_XCLBIN=") && p[0].ends_with("does not exist"));
        assert_eq!(p[1], "RLX_XDNA_INSTS is not set");
        assert!(p[2].contains("RLX_XDNA_GEMM=64,x,64"), "{p:?}");
        assert!(env_path_problem("RLX_XDNA_SHIM").is_none());
    }

    #[test]
    fn overlay_from_env_parses_complete_config() {
        let present = std::env::current_exe().unwrap();
        let present = present.to_str().unwrap();
        let _s = rlx_ir::env::scoped("RLX_XDNA_SHIM", present);
        let _x = rlx_ir::env::scoped("RLX_XDNA_XCLBIN", present);
        let _i = rlx_ir::env::scoped("RLX_XDNA_INSTS", present);
        let _g = rlx_ir::env::scoped("RLX_XDNA_GEMM", "64, 128,256");
        let ov = overlay_from_env().expect("complete overlay config");
        assert_eq!((ov.m, ov.k, ov.n), (64, 128, 256));
        assert_eq!(ov.xclbin, present);
        assert!(overlay_problems().is_empty());
    }

    #[test]
    fn memlock_soft_limit_parses_proc_limits() {
        let limits = "Limit                     Soft Limit           Hard Limit           Units\n\
                      Max open files            1024                 1048576              files\n\
                      Max locked memory         8388608              8388608              bytes\n";
        assert_eq!(memlock_soft_bytes(limits), Some(8 << 20));
        let unlimited = limits.replace(
            "8388608              8388608",
            "unlimited            unlimited",
        );
        assert_eq!(memlock_soft_bytes(&unlimited), None);
    }

    #[test]
    fn xdna2_is_eight_columns_or_more() {
        let g = |cols| AieGeometry {
            cols,
            rows: 6,
            col_size: 0,
            version: (1, 1),
        };
        assert!(!g(5).is_xdna2()); // Phoenix / Hawk Point
        assert!(g(8).is_xdna2()); // Strix
    }

    #[test]
    fn legacy_supported_ops_const_is_empty() {
        // Back-compat: the live op list is `XdnaBackend::supported_ops()` in
        // `rlx-runtime`; this legacy const stays empty.
        assert!(SUPPORTED_OPS.is_empty());
    }
}
