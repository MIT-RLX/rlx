// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! `cargo rlx doctor` — what runs on this host, what doesn't, and the fix.
//!
//! One report instead of a panic per missing piece: for every device the
//! binary knows, whether its backend is compiled in, whether the hardware and
//! driver are present (with the backend's own reason when not), and — for the
//! available ones — a small `relu(a·b + c)` graph executed on the device and
//! scored against an f64 reference computed here, not against rlx's CPU path.
//! The environment section lists the `RLX_*` variables that are set, flags the
//! ones the registry does not know (with the closest registered name), and
//! names deprecated spellings.
//!
//! The report is plain data so other front ends (pyrlx, the bindings) can
//! render it; [`std::fmt::Display`] is the CLI rendering.

use std::fmt;
use std::time::Instant;

use rlx_ir::env_registry::{self, EnvStability};
use rlx_ir::op::{Activation, BinaryOp};
use rlx_ir::{DType, Graph, Op, Shape};

use crate::Device;
use crate::device_ext::{feature_compiled, full_name, is_available, unavailable_reason};

pub mod facts;
pub mod known_issues;
mod report;

pub use facts::HostFacts;
pub use known_issues::{KnownIssue, Severity};

/// Options for [`doctor`].
#[derive(Debug, Clone, Default)]
pub struct DoctorOptions {
    /// Skip executing the check graph (availability + environment only).
    pub no_run: bool,
    /// Restrict the report to these devices (default: every device).
    pub devices: Option<Vec<Device>>,
}

/// Overall state of one device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceStatus {
    /// Available and the check graph matched the reference (or was skipped).
    Ready,
    /// Available, but the check graph failed or disagreed with the reference.
    Broken,
    /// Backend compiled in; hardware, driver or runtime missing.
    Unavailable,
    /// Backend not compiled into this binary.
    NotCompiled,
}

impl DeviceStatus {
    pub fn label(self) -> &'static str {
        match self {
            DeviceStatus::Ready => "ready",
            DeviceStatus::Broken => "BROKEN",
            DeviceStatus::Unavailable => "unavailable",
            DeviceStatus::NotCompiled => "not built",
        }
    }
}

/// Result of running the check graph on a device.
#[derive(Debug, Clone)]
pub struct RunCheck {
    /// `max |out - ref| / max |ref|`; `None` when the run failed.
    pub rel_err: Option<f64>,
    /// Tolerance `rel_err` was held to (looser on INT8 / f16 accelerators).
    pub tolerance: f64,
    /// Wall time of compile + first run, in milliseconds.
    pub millis: f64,
    /// Compile / run error or panic message.
    pub error: Option<String>,
}

impl RunCheck {
    pub fn passed(&self) -> bool {
        self.error.is_none() && self.rel_err.is_some_and(|e| e <= self.tolerance)
    }
}

/// One device's row.
#[derive(Debug, Clone)]
pub struct DeviceReport {
    pub device: Device,
    pub name: &'static str,
    /// Cargo feature that builds the backend.
    pub feature: &'static str,
    pub status: DeviceStatus,
    /// What to do, when the device is not ready.
    pub fix: Option<String>,
    pub check: Option<RunCheck>,
    /// Adapter name and external toolchain versions found on this host.
    pub detail: Option<String>,
    /// External toolchain versions this backend has been run against
    /// ([`tested_toolchain`]).
    pub tested: Option<String>,
    /// The detected toolchain is outside what rlx supports or knows.
    pub toolchain_warning: Option<String>,
}

/// One `RLX_*` variable set in the environment.
#[derive(Debug, Clone)]
pub struct EnvVarReport {
    pub name: String,
    pub value: String,
    pub status: EnvVarStatus,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvVarStatus {
    /// Registered; `group` from the registry.
    Known { group: &'static str },
    /// Registered as deprecated, or set via an alias.
    Deprecated { use_instead: String },
    /// Not in the registry — most likely a typo; read by nothing.
    Unknown { did_you_mean: Option<&'static str> },
}

/// The whole report.
#[derive(Debug, Clone)]
pub struct DoctorReport {
    pub version: &'static str,
    pub os: &'static str,
    pub arch: &'static str,
    pub devices: Vec<DeviceReport>,
    pub env: Vec<EnvVarReport>,
    /// Variables switched on by `RLX_PROFILE`, `RLX_CONFIG` or a family list:
    /// `(name, value, source)`.
    pub implied: Vec<(&'static str, String, String)>,
    /// Backend features compiled into this binary.
    pub features: Vec<&'static str>,
    /// Detected external versions (`cuda_driver`, `nvrtc`, `mlir_aie`).
    pub versions: Vec<(&'static str, (u32, u32, u32))>,
    /// wgpu sits on a Vulkan/DX12 adapter (host-routed elementwise).
    pub wgpu_host_routed: bool,
    /// Known issues that apply to this host, most severe first.
    pub issues: Vec<&'static KnownIssue>,
}

impl DoctorReport {
    /// No compiled-in, available device failed its check and no `RLX_*`
    /// variable is unknown. Drives the CLI exit code.
    pub fn healthy(&self) -> bool {
        !self.issues.iter().any(|i| i.severity == Severity::Error)
            && !self
                .devices
                .iter()
                .any(|d| d.status == DeviceStatus::Broken)
            && !self
                .env
                .iter()
                .any(|e| matches!(e.status, EnvVarStatus::Unknown { .. }))
    }

    pub fn ready_devices(&self) -> impl Iterator<Item = Device> + '_ {
        self.devices
            .iter()
            .filter(|d| d.status == DeviceStatus::Ready)
            .map(|d| d.device)
    }
}

/// Build the report. Never panics on a backend failure: compile errors and
/// panics inside the check run land in [`RunCheck::error`].
pub fn doctor(opts: &DoctorOptions) -> DoctorReport {
    let devices: Vec<Device> = match &opts.devices {
        Some(list) => list.clone(),
        None => Device::all().to_vec(),
    };
    let mut report = DoctorReport {
        version: env!("CARGO_PKG_VERSION"),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        devices: devices
            .into_iter()
            .map(|d| device_report(d, opts.no_run))
            .collect(),
        env: env_report(std::env::vars()),
        implied: rlx_ir::env::implied()
            .into_iter()
            .map(|(n, v, l)| (n, v, l.to_string()))
            .collect(),
        features: Device::all()
            .iter()
            .filter(|d| feature_compiled(**d))
            .map(|d| crate::session::feature_name(*d))
            .collect(),
        versions: detected_versions(),
        wgpu_host_routed: wgpu_host_routed(),
        issues: Vec::new(),
    };
    report.issues = known_issues::matching(&HostFacts::from_report(&report));
    report
}

/// External toolchain versions this binary can see.
fn detected_versions() -> Vec<(&'static str, (u32, u32, u32))> {
    #[allow(unused_mut)]
    let mut v: Vec<(&'static str, (u32, u32, u32))> = Vec::new();
    #[cfg(feature = "cuda")]
    {
        if let Some((a, b)) = rlx_cuda::device::driver_version() {
            v.push(("cuda_driver", (a, b, 0)));
        }
        if let Some((a, b)) = rlx_cuda::device::nvrtc_version() {
            v.push(("nvrtc", (a, b, 0)));
        }
    }
    #[cfg(feature = "xdna")]
    if let Some(ver) = rlx_xdna::toolchain::toolchain()
        .ok()
        .and_then(|t| t.version)
    {
        v.push(("mlir_aie", ver));
    }
    v
}

fn wgpu_host_routed() -> bool {
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    {
        is_available(Device::Gpu) && rlx_wgpu::device::coop_discrete_backend()
    }
    #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
    {
        false
    }
}

fn device_report(device: Device, no_run: bool) -> DeviceReport {
    let feature = crate::session::feature_name(device);
    let base = |status, fix| DeviceReport {
        device,
        name: full_name(device),
        feature,
        status,
        fix,
        check: None,
        detail: None,
        tested: tested_toolchain(device),
        toolchain_warning: None,
    };
    if !feature_compiled(device) {
        return base(
            DeviceStatus::NotCompiled,
            Some(format!("build with `--features {feature}`")),
        );
    }
    let (detail, toolchain_warning) = toolchain_probe(device);
    let with_toolchain = |mut r: DeviceReport| {
        r.detail = detail.clone();
        r.toolchain_warning = toolchain_warning.clone();
        r
    };
    if !is_available(device) {
        let why = unavailable_reason(device)
            .unwrap_or_else(|| "no such device or driver was found on this host".into());
        return with_toolchain(base(DeviceStatus::Unavailable, Some(why)));
    }
    if no_run {
        return with_toolchain(base(DeviceStatus::Ready, None));
    }
    let check = run_check(device);
    let mut r = with_toolchain(base(DeviceStatus::Ready, None));
    if !check.passed() {
        r.status = DeviceStatus::Broken;
        r.fix = Some(match &check.error {
            Some(e) => format!("check graph failed: {e}"),
            None => format!(
                "check graph disagrees with the f64 reference (rel err {:.2e} > {:.0e}); \
                 rerun with RLX_DISPATCH_REPORT=1 to see the dispatch",
                check.rel_err.unwrap_or(f64::NAN),
                check.tolerance
            ),
        });
    }
    r.check = Some(check);
    r
}

/// Rows of `tested_toolchains.tsv`: what the version matrix has actually run.
const TESTED_TSV: &str = include_str!("tested_toolchains.tsv");

/// One passing (component, version, hardware) the matrix ran for `device`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TestedRow {
    pub device: &'static str,
    pub component: &'static str,
    pub version: &'static str,
    pub hardware: &'static str,
}

/// Every passing row of `tested_toolchains.tsv`.
pub fn tested_rows() -> Vec<TestedRow> {
    TESTED_TSV
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|l| {
            let c: Vec<&'static str> = l.split('\t').collect();
            (c.len() >= 5 && c[4] == "pass").then(|| TestedRow {
                device: c[0],
                component: c[1],
                version: c[2],
                hardware: c[3],
            })
        })
        .collect()
}

/// External toolchain versions `device` has actually been run against, from
/// the version matrix (`tested_toolchains.tsv`), e.g. `mlir-aie 1.3.4, 1.4.4;
/// xrt 2.21.75 on Ryzen 9 8945HS npu1`. `None` when nothing was recorded.
pub fn tested_toolchain(device: Device) -> Option<String> {
    let key = crate::session::feature_name(device);
    let rows: Vec<TestedRow> = tested_rows()
        .into_iter()
        .filter(|r| r.device == key)
        .collect();
    if rows.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    let mut components: Vec<&str> = rows.iter().map(|r| r.component).collect();
    components.dedup();
    for c in components {
        let mut vs: Vec<&str> = rows
            .iter()
            .filter(|r| r.component == c)
            .map(|r| r.version)
            .collect();
        vs.dedup();
        parts.push(format!("{c} {}", vs.join(", ")));
    }
    let mut hw: Vec<&str> = rows.iter().map(|r| r.hardware).collect();
    hw.sort_unstable();
    hw.dedup();
    Some(format!("{} on {}", parts.join("; "), hw.join(" / ")))
}

/// Adapter / toolchain versions for `device`, plus a warning when they fall
/// outside [`tested_toolchain`]'s supported range. Cheap: reuses the contexts
/// the availability probe already opened.
fn toolchain_probe(device: Device) -> (Option<String>, Option<String>) {
    let _ = device;
    #[cfg(feature = "cuda")]
    if device == Device::Cuda {
        let fmt = |v: Option<(u32, u32)>| {
            v.map(|(a, b)| format!("{a}.{b}"))
                .unwrap_or_else(|| "not found".into())
        };
        let driver = rlx_cuda::device::driver_version();
        let detail = format!(
            "driver CUDA {}, NVRTC {}",
            fmt(driver),
            fmt(rlx_cuda::device::nvrtc_version())
        );
        let warn = driver.filter(|v| *v < (12, 3)).map(|(a, b)| {
            format!(
                "driver supports CUDA {a}.{b}; rlx-cuda needs >= 12.3 — update the NVIDIA driver"
            )
        });
        return (Some(detail), warn);
    }
    #[cfg(feature = "vulkan")]
    if device == Device::Vulkan {
        return (rlx_vulkan::device_name(), None);
    }
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    if device == Device::Gpu {
        return (rlx_wgpu::device::adapter_name(), None);
    }
    #[cfg(all(feature = "mlx", rlx_mlx_host))]
    if device == Device::Mlx {
        return (Some(format!("MLX {}", rlx_mlx::array::version())), None);
    }
    #[cfg(feature = "xdna")]
    if device == Device::Xdna {
        // Newer-than-known mlir-aie is reported as known issue RLX-XDNA-001.
        return match rlx_xdna::toolchain::toolchain() {
            Ok(tc) => {
                let how = match tc.source {
                    rlx_xdna::toolchain::Source::Env => "from AIECC",
                    rlx_xdna::toolchain::Source::Discovered(_) => "found",
                };
                let ver = tc
                    .version
                    .map(|(a, b, c)| format!("mlir-aie {a}.{b}.{c}"))
                    .unwrap_or_else(|| "mlir-aie (version unknown, source build?)".into());
                let warn = tc.peano_mismatch().map(|(pinned, installed)| {
                    format!(
                        "{ver} pins Peano {pinned}, found {installed} — if kernels fail \
                         to compile, install the pinned build (RLX-XDNA-010)"
                    )
                });
                (Some(format!("{ver} {how} at {}", tc.aiecc.display())), warn)
            }
            Err(_) => (None, None),
        };
    }
    (None, None)
}

const M: usize = 4;
const K: usize = 8;
const N: usize = 4;

/// Deterministic, sign-mixed operands with small exact-in-f32 values.
fn check_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let v = |len: usize, seed: usize| -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 7 + seed * 13) % 17) as f32 - 8.0) / 8.0)
            .collect()
    };
    (v(M * K, 1), v(K * N, 2), v(M * N, 3))
}

fn check_graph() -> Graph {
    let mut g = Graph::new("rlx_doctor");
    let a = g.input("a", Shape::new(&[M, K], DType::F32));
    let b = g.input("b", Shape::new(&[K, N], DType::F32));
    let c = g.input("c", Shape::new(&[M, N], DType::F32));
    let mn = Shape::new(&[M, N], DType::F32);
    let ab = g.add_node(Op::MatMul, vec![a, b], mn.clone());
    let s = g.add_node(Op::Binary(BinaryOp::Add), vec![ab, c], mn.clone());
    let y = g.add_node(Op::Activation(Activation::Relu), vec![s], mn);
    g.set_outputs(vec![y]);
    g
}

/// `relu(a·b + c)` in f64.
fn reference(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0f64; M * N];
    for i in 0..M {
        for j in 0..N {
            let dot: f64 = (0..K)
                .map(|k| a[i * K + k] as f64 * b[k * N + j] as f64)
                .sum();
            out[i * N + j] = (dot + c[i * N + j] as f64).max(0.0);
        }
    }
    out
}

/// Accelerators that compute the check in INT8 or f16 by design.
fn tolerance(device: Device) -> f64 {
    match device {
        Device::Xdna | Device::Hexagon | Device::Ane => 5e-2,
        Device::Tpu => 1e-2,
        _ => 1e-4,
    }
}

fn run_check(device: Device) -> RunCheck {
    let (a, b, c) = check_inputs();
    let want = reference(&a, &b, &c);
    let tol = tolerance(device);
    let t0 = Instant::now();
    let result = quiet_catch(|| -> Result<Vec<f32>, String> {
        let session = crate::Session::try_new(device).map_err(|e| e.to_string())?;
        let mut compiled = session
            .try_compile(check_graph())
            .map_err(|e| e.to_string())?;
        let mut outs = compiled
            .try_run(&[("a", &a), ("b", &b), ("c", &c)])
            .map_err(|e| e.to_string())?;
        outs.pop().ok_or_else(|| "no output".to_string())
    });
    let millis = t0.elapsed().as_secs_f64() * 1e3;
    match result {
        Ok(got) if got.len() == want.len() => {
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-6);
            let err = got
                .iter()
                .zip(&want)
                .map(|(g, w)| (*g as f64 - w).abs())
                .fold(0.0f64, |m, e| if e.is_nan() { f64::NAN } else { m.max(e) });
            RunCheck {
                rel_err: Some(err / scale),
                tolerance: tol,
                millis,
                error: None,
            }
        }
        Ok(got) => RunCheck {
            rel_err: None,
            tolerance: tol,
            millis,
            error: Some(format!("expected {} values, got {}", want.len(), got.len())),
        },
        Err(e) => RunCheck {
            rel_err: None,
            tolerance: tol,
            millis,
            error: Some(e),
        },
    }
}

/// Run `f`, turning a panic into `Err(message)` without printing it.
fn quiet_catch<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    std::panic::set_hook(prev);
    match r {
        Ok(r) => r,
        Err(p) => Err(p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .map(|m| format!("panicked: {m}"))
            .unwrap_or_else(|| "panicked".into())),
    }
}

/// Classify every `RLX_*` pair in `vars`, sorted by name.
pub fn env_report(vars: impl IntoIterator<Item = (String, String)>) -> Vec<EnvVarReport> {
    let mut out: Vec<EnvVarReport> = vars
        .into_iter()
        .filter(|(k, _)| k.starts_with("RLX_"))
        .map(|(name, value)| {
            let status = match env_registry::lookup(&name) {
                Some(e) => match e.stability {
                    EnvStability::Deprecated { replace_with } => EnvVarStatus::Deprecated {
                        use_instead: replace_with.to_string(),
                    },
                    _ if e.name != name => EnvVarStatus::Deprecated {
                        use_instead: e.name.to_string(),
                    },
                    _ => EnvVarStatus::Known { group: e.group },
                },
                None => EnvVarStatus::Unknown {
                    did_you_mean: env_registry::suggest(&name),
                },
            };
            EnvVarReport {
                name,
                value,
                status,
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

impl fmt::Display for DoctorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "rlx {} on {}-{}", self.version, self.os, self.arch)?;
        writeln!(f)?;
        writeln!(f, "Devices")?;
        // Built-in devices first; the not-built ones collapse to one line.
        let mut not_built: Vec<&str> = Vec::new();
        for d in &self.devices {
            if d.status == DeviceStatus::NotCompiled {
                not_built.push(d.feature);
                continue;
            }
            let mark = match d.status {
                DeviceStatus::Ready => "ok ",
                DeviceStatus::Broken => "ERR",
                _ => " - ",
            };
            let label = d.device.to_string();
            write!(f, "  {mark} {label:<11} {:<12}", d.status.label())?;
            // The full name only when it adds something (`CPU (Accelerate)`).
            if d.name != label {
                write!(f, "{}  ", d.name)?;
            }
            match &d.check {
                Some(c) if c.passed() => writeln!(
                    f,
                    "check rel err {:.1e} in {:.0} ms",
                    c.rel_err.unwrap_or(0.0),
                    c.millis
                )?,
                _ => writeln!(f)?,
            }
            if let Some(detail) = &d.detail {
                writeln!(f, "        {detail}")?;
            }
            if let Some(w) = &d.toolchain_warning {
                for (i, line) in wrap(w, 84).into_iter().enumerate() {
                    let lead = if i == 0 { "warning: " } else { "         " };
                    writeln!(f, "        {lead}{line}")?;
                }
            }
            if let Some(fix) = &d.fix {
                for line in wrap(fix, 88) {
                    writeln!(f, "        {line}")?;
                }
            }
        }
        if !not_built.is_empty() {
            writeln!(
                f,
                "  not built into this binary (enable the Cargo feature): {}",
                not_built.join(", ")
            )?;
        }
        writeln!(f)?;
        if self.env.is_empty() {
            writeln!(f, "Environment: no RLX_* variables set")?;
        } else {
            writeln!(f, "Environment ({} RLX_* set)", self.env.len())?;
            for e in &self.env {
                match &e.status {
                    EnvVarStatus::Known { group } => {
                        writeln!(f, "  ok  {}={}  [{group}]", e.name, e.value)?
                    }
                    EnvVarStatus::Deprecated { use_instead } => writeln!(
                        f,
                        "  old {}={}  — deprecated, use {use_instead}",
                        e.name, e.value
                    )?,
                    EnvVarStatus::Unknown { did_you_mean } => writeln!(
                        f,
                        "  ??  {}={}  — not a known variable, nothing reads it{}",
                        e.name,
                        e.value,
                        did_you_mean
                            .map(|s| format!("; did you mean {s}?"))
                            .unwrap_or_default()
                    )?,
                }
            }
        }
        if !self.implied.is_empty() {
            writeln!(
                f,
                "  set by profile / config / list ({}):",
                self.implied.len()
            )?;
            for (name, value, source) in &self.implied {
                writeln!(f, "      {name}={value}  ← {source}")?;
            }
        }
        if !self.issues.is_empty() {
            writeln!(f)?;
            writeln!(f, "Known issues on this host (docs/known-issues.md)")?;
            for i in &self.issues {
                writeln!(f, "  {:<7} {}  {}", i.severity.label(), i.id, i.title)?;
                for (n, line) in wrap(&format!("fix: {}", i.fix), 86).into_iter().enumerate() {
                    writeln!(f, "        {}{line}", if n == 0 { "" } else { "     " })?;
                }
            }
        }
        writeln!(f)?;
        let ready: Vec<String> = self.ready_devices().map(|d| d.to_string()).collect();
        if self.healthy() {
            write!(f, "healthy — ready: {}", ready.join(", "))
        } else {
            write!(f, "problems found — ready: {}", ready.join(", "))
        }
    }
}

/// Greedy word wrap; backend diagnostics are one long sentence.
fn wrap(s: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    for para in s.lines() {
        let mut line = String::new();
        for word in para.split_whitespace() {
            if !line.is_empty() && line.len() + 1 + word.len() > width {
                lines.push(std::mem::take(&mut line));
            }
            if !line.is_empty() {
                line.push(' ');
            }
            line.push_str(word);
        }
        if !line.is_empty() {
            lines.push(line);
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_matches_hand_computation() {
        let (a, b, c) = check_inputs();
        let r = reference(&a, &b, &c);
        let (i, j) = (1, 2);
        let dot: f64 = (0..K).map(|k| (a[i * K + k] * b[k * N + j]) as f64).sum();
        assert_eq!(r[i * N + j], (dot + c[i * N + j] as f64).max(0.0));
        // Sign-mixed inputs: relu must both clamp and pass something.
        assert!(r.contains(&0.0) && r.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn typo_gets_a_suggestion_and_known_names_classify() {
        let rep = env_report([
            ("RLX_DISPATCH_REPOT".to_string(), "1".to_string()), // env-lint: ignore
            ("RLX_DISPATCH_REPORT".to_string(), "1".to_string()),
            ("RLX_TOTALLY_MADE_UP_NAME_XYZ".to_string(), "1".to_string()), // env-lint: ignore
            ("PATH".to_string(), "/bin".to_string()),
        ]);
        assert_eq!(rep.len(), 3, "non-RLX vars are ignored");
        let by = |n: &str| rep.iter().find(|e| e.name == n).unwrap().status.clone();
        assert!(matches!(
            by("RLX_DISPATCH_REPORT"),
            EnvVarStatus::Known { .. }
        ));
        assert_eq!(
            by("RLX_DISPATCH_REPOT"), // env-lint: ignore
            EnvVarStatus::Unknown {
                did_you_mean: Some("RLX_DISPATCH_REPORT")
            }
        );
        assert_eq!(
            by("RLX_TOTALLY_MADE_UP_NAME_XYZ"), // env-lint: ignore
            EnvVarStatus::Unknown { did_you_mean: None }
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn cpu_passes_its_own_check() {
        let rep = doctor(&DoctorOptions {
            no_run: false,
            devices: Some(vec![Device::Cpu]),
        });
        let cpu = &rep.devices[0];
        assert_eq!(cpu.status, DeviceStatus::Ready, "{:?}", cpu);
        assert!(cpu.check.as_ref().unwrap().passed());
        let text = rep.to_string();
        assert!(text.contains("ready"), "{text}");
    }

    #[test]
    fn uncompiled_device_names_its_feature() {
        let missing = Device::all()
            .iter()
            .copied()
            .find(|d| !feature_compiled(*d));
        if let Some(d) = missing {
            let r = device_report(d, true);
            assert_eq!(r.status, DeviceStatus::NotCompiled);
            assert!(r.fix.unwrap().contains(r.feature));
        }
    }
}
