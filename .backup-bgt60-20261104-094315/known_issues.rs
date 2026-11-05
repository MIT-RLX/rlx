// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Problems users have hit, how to recognize each, and the fix.
//!
//! Every resolved GitHub issue becomes a row here, so the next user on the
//! same setup gets the answer from `cargo rlx doctor` instead of filing a new
//! issue. An entry with a [`KnownIssue::detect`] check is diagnosed
//! automatically from [`HostFacts`]; one without (build-time failures, which
//! happen before `doctor` can run) is listed by `cargo rlx doctor --known` and
//! in `docs/known-issues.md` so its symptom text is searchable.
//!
//! Every `detect` check has a fake-host test below, and `docs/known-issues.md`
//! is generated from this table (`just gen-known-issues`, checked by a test).

use super::DeviceStatus;
use super::facts::HostFacts;
use crate::Device;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// Blocks a device or gives wrong results.
    Error,
    /// Works, but slower than it should or one step from breaking.
    Warning,
    /// Expected behaviour worth knowing about.
    Info,
}

impl Severity {
    pub fn label(self) -> &'static str {
        match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        }
    }
}

#[derive(Debug)]
pub struct KnownIssue {
    /// Stable id, `RLX-<AREA>-<NNN>`. Never reused.
    pub id: &'static str,
    pub title: &'static str,
    pub severity: Severity,
    /// What the user sees: an error message fragment to search for.
    pub symptom: &'static str,
    pub cause: &'static str,
    /// What to do — commands where there are commands.
    pub fix: &'static str,
    /// Diagnose from host facts; `None` for build-time issues.
    pub detect: Option<fn(&HostFacts) -> bool>,
}

/// The table. Append only; ids are permanent references from issues and docs.
pub static ISSUES: &[KnownIssue] = &[
    // ── XDNA ────────────────────────────────────────────────────────────────
    KnownIssue {
        id: "RLX-XDNA-001",
        title: "mlir-aie is newer than any release rlx has been run against",
        severity: Severity::Warning,
        symptom: "aiecc: ... error: expected ')' / unknown type in dialect `aie`",
        cause: "mlir-aie changes its AIE-MLIR syntax between releases (1.4.0 dma_bd, \
                1.4.3 objectfifosubview). rlx rewrites its kernels for every release \
                up to 1.4.x; a newer one may have changed something else.",
        fix: "Run `cargo rlx doctor --report` and open an issue with it. Meanwhile pin a \
              known release: `pip install mlir_aie==1.4.4`, or set \
              RLX_XDNA_AIE_DIALECT=1.4.4 to force the newest known syntax.",
        detect: Some(|f| f.version("mlir_aie").is_some_and(|v| v >= (1, 5, 0))),
    },
    KnownIssue {
        id: "RLX-XDNA-002",
        title: "rlx's XRT shim cannot be built: XRT headers missing",
        severity: Severity::Error,
        symptom: "building it needs the XRT headers (xrt/xrt_device.h)",
        cause: "RLX_XDNA_SHIM is unset, so rlx builds librlx_xdna_shim.so itself, and \
                that needs XRT's development headers, which the runtime packages do \
                not ship.",
        fix: "Ubuntu/Debian: `sudo apt install libxrt-dev`. Or set XILINX_XRT to an XRT \
              tree that has include/xrt/, or build the shim yourself and set RLX_XDNA_SHIM.",
        detect: Some(|f| f.reason_contains(Device::Xdna, "XRT headers")),
    },
    KnownIssue {
        id: "RLX-XDNA-003",
        title: "XDNA NPU present but no XRT runtime",
        severity: Severity::Error,
        symptom: "AMD XDNA NPU detected ... but no userspace runtime is available",
        cause: "The amdxdna kernel driver is loaded, but the XRT userspace libraries \
                (libxrt_coreutil, libxrt_driver_xdna) are not found.",
        fix: "Ubuntu 26.04: `sudo apt install libxrt2 libxrt-npu2 libxrt-utils-npu`. \
              Without root: `apt-get download` those packages, `dpkg-deb -x` them into a \
              directory, and set RLX_XDNA_XRT_LIB to its usr/lib/x86_64-linux-gnu. \
              Check with `xrt-smi examine`.",
        detect: Some(|f| f.reason_contains(Device::Xdna, "no userspace runtime")),
    },
    KnownIssue {
        id: "RLX-XDNA-004",
        title: "NPU device node is not writable by this user",
        severity: Severity::Error,
        symptom: "/dev/accel/accel0 is not read/write for this user",
        cause: "XRT opens /dev/accel/accelN read/write; the user is not in its group.",
        fix: "`sudo usermod -aG render $USER`, then log out and back in.",
        detect: Some(|f| f.reason_contains(Device::Xdna, "is not read/write")),
    },
    KnownIssue {
        id: "RLX-XDNA-005",
        title: "Locked-memory limit too low for XRT buffers",
        severity: Severity::Warning,
        symptom: "XRT buffer mmap fails with EAGAIN",
        cause: "The distro default RLIMIT_MEMLOCK (8 MiB) is smaller than larger NPU buffers.",
        fix: "Add `@render - memlock unlimited` to /etc/security/limits.d/rlx.conf, then \
              log in again.",
        detect: Some(|f| f.reason_contains(Device::Xdna, "RLIMIT_MEMLOCK")),
    },
    KnownIssue {
        id: "RLX-XDNA-006",
        title: "XDNA 2 (Strix-class) NPU is not supported yet",
        severity: Severity::Error,
        symptom: "this is an XDNA 2 (Strix-class, AIE2P) NPU",
        cause: "rlx-xdna's kernels target XDNA 1 (Phoenix / Hawk Point npu1); an npu1 \
                xclbin cannot load on AIE2P.",
        fix: "No workaround yet. Use Device::Rocm or Device::Vulkan on the same machine; \
              follow the XDNA 2 tracking issue.",
        detect: Some(|f| f.reason_contains(Device::Xdna, "XDNA 2")),
    },
    KnownIssue {
        id: "RLX-XDNA-007",
        title: "mlir-aie toolchain (aiecc + Peano) not found",
        severity: Severity::Error,
        symptom: "no mlir-aie toolchain found",
        cause: "On-demand NPU kernels are compiled with mlir-aie's aiecc and the Peano \
                (llvm-aie) core compiler; neither was found in PATH, ~/.local, venvs \
                under $HOME, /usr or /opt.",
        fix: "`cargo rlx setup xdna` installs mlir-aie 1.4.4 with its pinned Peano into \
              ~/.local/share/rlx/toolchains (no root). Any venv also works and is found \
              automatically; or set AIECC and PEANO, or RLX_XDNA_TOOLCHAIN_DIR.",
        detect: Some(|f| {
            f.reason_contains(Device::Xdna, "mlir-aie toolchain found")
                || f.reason_contains(Device::Xdna, "no Peano")
        }),
    },
    KnownIssue {
        id: "RLX-XDNA-008",
        title: "XRT on LD_LIBRARY_PATH hides ROCm's HIP",
        severity: Severity::Warning,
        symptom: "Device::Rocm unavailable after adding XRT to LD_LIBRARY_PATH",
        cause: "The extracted XRT tree ships its own libxrt_hip.so; first on \
                LD_LIBRARY_PATH, it shadows ROCm's HIP runtime.",
        fix: "Remove the XRT directory from LD_LIBRARY_PATH and set RLX_XDNA_XRT_LIB to it \
              instead — rlx loads XRT from there without affecting other libraries.",
        detect: Some(|f| {
            f.compiled(Device::Rocm)
                && f.env_dirs("LD_LIBRARY_PATH")
                    .iter()
                    .any(|d| f.exists(&format!("{d}/libxrt_hip.so")))
        }),
    },
    KnownIssue {
        id: "RLX-XDNA-009",
        title: "mlir-aie older than 1.3 has no native aiecc",
        severity: Severity::Error,
        symptom: "ships only aiecc.py",
        cause: "Releases before 1.3.0 ship the Python driver aiecc.py only; rlx drives \
                the native aiecc binary (no Python in the loop).",
        fix: "Install mlir-aie 1.3.0 or newer (1.4.4 recommended) together with its \
              pinned Peano: `pip install mlir_aie==1.4.4 -f \
              https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4` and \
              `pip install -r utils/peano-requirements.txt` from that release.",
        detect: Some(|f| f.reason_contains(Device::Xdna, "ships only aiecc.py")),
    },
    KnownIssue {
        id: "RLX-XDNA-010",
        title: "Peano (llvm-aie) is not the build this mlir-aie release pins",
        severity: Severity::Warning,
        symptom: "llvm-aie/bin/opt: …peano-compat.ll: error: expected value token",
        cause: "Each mlir-aie release pins one llvm-aie build \
                (utils/peano-requirements.txt); LLVM IR from one can fail to parse in \
                another's opt. Some mismatches work (1.4.4 runs on the 1.4.0 Peano).",
        fix: "Install the pinned build next to mlir-aie: `pip install -r \
              utils/peano-requirements.txt` from the matching mlir-aie tag (e.g. 1.4.4 → \
              `llvm-aie==22.0.0.2026092101 -f \
              https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly`).",
        detect: Some(|f| f.reason_contains(Device::Xdna, "pins Peano")),
    },
    KnownIssue {
        id: "RLX-XDNA-011",
        title: "mlir-aie 1.3.0, 1.3.1 and 1.3.3 cannot compile every rlx kernel correctly",
        severity: Severity::Error,
        symptom: "cannot compile every rlx kernel correctly / ran out of registers / expected value token (f0x… literal)",
        cause: "Measured on the NPU rig against every published Peano build: on 1.3.0 \
                softmax and attention compile but return wrong results, 1.3.1 runs out \
                of registers on them, and 1.3.3 emits LLVM IR float literals (f0x…) that \
                no Peano parses. rlx refuses these releases (XDNA reports unavailable) \
                unless RLX_XDNA_ALLOW_KNOWN_BAD=1. 1.3.2, 1.3.4 and 1.4.0-1.4.4 pass the \
                full suite.",
        fix: "`cargo rlx setup xdna` (installs 1.4.4 with its pinned Peano), or install \
              1.4.4 by hand: `pip install mlir_aie==1.4.4 -f \
              https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4` and \
              `pip install -r utils/peano-requirements.txt` from that release.",
        detect: Some(|f| {
            matches!(
                f.version("mlir_aie"),
                Some((1, 3, 0)) | Some((1, 3, 1)) | Some((1, 3, 3))
            ) || f.reason_contains(Device::Xdna, "RLX-XDNA-011")
        }),
    },
    // ── ROCm ────────────────────────────────────────────────────────────────
    KnownIssue {
        id: "RLX-ROCM-001",
        title: "MIOpen has no kernel database",
        severity: Severity::Info,
        symptom: "MIOpen has no kernel database",
        cause: "Distro MIOpen builds (+dfsg) ship without the precompiled kernel \
                database, so rlx uses its own convolution kernels instead.",
        fix: "For MIOpen convolutions, install AMD's packaged MIOpen (repo.radeon.com) or \
              set MIOPEN_SYSTEM_DB_PATH to a directory holding its *.kdb files.",
        detect: Some(|f| {
            f.compiled(Device::Rocm) && {
                let mut roots: Vec<String> = Vec::new();
                if let Some(d) = f.env("MIOPEN_SYSTEM_DB_PATH") {
                    roots.push(d.to_string());
                }
                if let Some(r) = f.env("ROCM_PATH") {
                    roots.push(format!("{r}/share/miopen/db"));
                }
                roots.push("/opt/rocm/share/miopen/db".into());
                !roots.iter().any(|r| f.dir_nonempty(r))
            }
        }),
    },
    KnownIssue {
        id: "RLX-ROCM-002",
        title: "Radeon 780M (gfx1103) is not enumerated by ROCm",
        severity: Severity::Error,
        symptom: "no ROCm-capable device is detected",
        cause: "ROCm does not list gfx1103 as supported; HSA hides it unless told to treat \
                it as gfx1100.",
        fix: "`export HSA_OVERRIDE_GFX_VERSION=11.0.0` before running (hipRTC then compiles \
              for gfx1100, which runs on the 780M).",
        detect: Some(|f| {
            f.status(Device::Rocm) == Some(DeviceStatus::Unavailable)
                && f.env("HSA_OVERRIDE_GFX_VERSION").is_none()
                && f.any_child_file_contains(
                    "/sys/class/kfd/kfd/topology/nodes",
                    "properties",
                    "gfx_target_version 110003",
                )
        }),
    },
    KnownIssue {
        id: "RLX-ROCM-003",
        title: "APU GPU resets under NUMA balancing (gfx11 MES queue-eviction hang)",
        severity: Severity::Info,
        symptom: "HipError(719); kernel log: MES failed to respond to msg=REMOVE_QUEUE … \
                  Failed to evict process queues … GPU reset",
        cause: "On a multi-node host with kernel.numa_balancing on, the kernel's page \
                sampling makes amdgpu evict a process's GPU queues, and gfx11 APU firmware \
                (MES) can hang doing it.",
        fix: "rlx-rocm opts its process out (MPOL_LOCAL, before hipInit). Threads started \
              before ROCm comes up keep the default: run under `numactl --localalloc`, or \
              `sudo sysctl kernel.numa_balancing=0`. RLX_ROCM_NUMA_GUARD=0 turns the opt-out off.",
        detect: Some(|f| {
            f.status(Device::Rocm) == Some(DeviceStatus::Ready)
                && std::fs::read_to_string(f.path("/proc/sys/kernel/numa_balancing"))
                    .is_ok_and(|v| v.trim() != "0")
                && f.exists("/sys/devices/system/node/node1")
                && f.any_child_file_contains(
                    "/sys/class/kfd/kfd/topology/nodes",
                    "properties",
                    "local_mem_size 0\n",
                )
        }),
    },
    // ── CUDA ────────────────────────────────────────────────────────────────
    KnownIssue {
        id: "RLX-CUDA-001",
        title: "NVIDIA driver older than CUDA 12.3",
        severity: Severity::Error,
        symptom: "GetProcAddress / symbol not found loading libcuda",
        cause: "rlx-cuda resolves the CUDA 12.3 driver API; older drivers lack symbols.",
        fix: "Update the NVIDIA driver (`nvidia-smi` shows the CUDA version it supports).",
        detect: Some(|f| f.version("cuda_driver").is_some_and(|v| v < (12, 3, 0))),
    },
    KnownIssue {
        id: "RLX-CUDA-002",
        title: "NVRTC (CUDA runtime compiler) not found",
        severity: Severity::Error,
        symptom: "libnvrtc.so: cannot open shared object file",
        cause: "rlx-cuda JIT-compiles its kernels with NVRTC; the driver is present but \
                libnvrtc is not on the library path.",
        fix: "Install it (Ubuntu: `sudo apt install cuda-nvrtc-13-1`, or the CUDA toolkit) \
              and make it findable: `export LD_LIBRARY_PATH=/usr/local/cuda/lib64`.",
        detect: Some(|f| {
            (f.version("cuda_driver").is_some() && f.version("nvrtc").is_none())
                || f.reason_contains(Device::Cuda, "nvrtc")
        }),
    },
    KnownIssue {
        id: "RLX-CUDA-003",
        title: "CUDA backend built, but no NVIDIA driver loaded",
        severity: Severity::Error,
        symptom: "device CUDA is not available",
        cause: "No /dev/nvidia0: the NVIDIA kernel driver is not loaded (or no NVIDIA GPU).",
        fix: "Install the NVIDIA driver and check with `nvidia-smi`.",
        detect: Some(|f| {
            f.status(Device::Cuda) == Some(DeviceStatus::Unavailable) && !f.exists("/dev/nvidia0")
        }),
    },
    // ── wgpu / Vulkan ───────────────────────────────────────────────────────
    KnownIssue {
        id: "RLX-WGPU-001",
        title: "wgpu on Vulkan/DX12 runs elementwise ops on the host",
        severity: Severity::Info,
        symptom: "most wgpu schedule steps are host_op (RLX_DISPATCH_REPORT=1)",
        cause: "Deliberate: elementwise SPIR-V produced wrong model output on discrete \
                NVIDIA, so wgpu host-routes those ops on every Vulkan/DX12 adapter \
                (integrated ones too). Results are correct but slower.",
        fix: "Prefer Device::Cuda or Device::Vulkan on this GPU. RLX_WGPU_SHARD_GPU=1 forces \
              GPU elementwise if you want to measure it.",
        detect: Some(|f| f.wgpu_host_routed),
    },
    KnownIssue {
        id: "RLX-VULKAN-001",
        title: "Vulkan backend built, but no Vulkan driver (ICD) installed",
        severity: Severity::Error,
        symptom: "device Vulkan is not available",
        cause: "No Vulkan ICD manifest in /usr/share/vulkan/icd.d or /etc/vulkan/icd.d.",
        fix: "AMD/Intel: `sudo apt install mesa-vulkan-drivers`; NVIDIA: the proprietary \
              driver ships its ICD. Check with `vulkaninfo --summary`.",
        detect: Some(|f| {
            f.status(Device::Vulkan) == Some(DeviceStatus::Unavailable)
                && !f.dir_nonempty("/usr/share/vulkan/icd.d")
                && !f.dir_nonempty("/etc/vulkan/icd.d")
        }),
    },
    // ── Environment ─────────────────────────────────────────────────────────
    KnownIssue {
        id: "RLX-ENV-001",
        title: "An RLX_* variable is set that nothing reads",
        severity: Severity::Warning,
        symptom: "RLX_... is not a variable rlx reads",
        cause: "Most often a typo; the setting silently does nothing.",
        fix: "Fix the name (`cargo rlx doctor` suggests the nearest one) or unset it.",
        detect: Some(|f| {
            f.env
                .iter()
                .any(|(k, _)| k.starts_with("RLX_") && !rlx_ir::env_registry::is_registered(k))
        }),
    },
    KnownIssue {
        id: "RLX-CHECK-001",
        title: "A device failed its check run",
        severity: Severity::Error,
        symptom: "BROKEN … check graph failed / disagrees with the f64 reference",
        cause: "The device is available but returned an error or wrong numbers for a \
                small relu(a·b + c) graph.",
        fix: "Rerun with RLX_DISPATCH_REPORT=1 to see what ran where, then open an issue \
              with `cargo rlx doctor --report`.",
        detect: Some(|f| f.devices.iter().any(|d| d.status == DeviceStatus::Broken)),
    },
    // ── Build time (listed, not detected: they stop the build before doctor runs) ──
    KnownIssue {
        id: "RLX-BUILD-001",
        title: "cargo older than 1.99 rejects the workspace manifests",
        severity: Severity::Error,
        symptom: "`default-features = false` cannot override workspace's `default-features`",
        cause: "Workspace dependency overrides that cargo 1.99 accepts.",
        fix: "`rustup update stable` (cargo >= 1.99).",
        detect: None,
    },
    KnownIssue {
        id: "RLX-BUILD-002",
        title: "Linking libpython fails in a cargo build",
        severity: Severity::Error,
        symptom: "unable to find library -lpython3",
        cause: "A PyO3 build without `extension-module` links libpython, whose unversioned \
                .so comes only with the -dev package.",
        fix: "Fixed in pyrlx (extension-module is on by default). Elsewhere: \
              `sudo apt install python3-dev`, or build Python packages with maturin.",
        detect: None,
    },
    KnownIssue {
        id: "RLX-BUILD-003",
        title: "OpenBLAS not installed",
        severity: Severity::Error,
        symptom: "unable to find library -lopenblas",
        cause: "rlx-cpu links OpenBLAS by default on Linux x86-64.",
        fix: "`sudo apt install libopenblas-dev`, or build rlx-cpu with \
              `--no-default-features --features parity-gemm` (pure Rust GEMM).",
        detect: None,
    },
    KnownIssue {
        id: "RLX-BUILD-004",
        title: "MLX submodule not checked out",
        severity: Severity::Error,
        symptom: "rlx-mlx-sys: vendor/mlx is empty",
        cause: "crates/backends/rlx-mlx-sys/vendor/mlx is a git submodule.",
        fix: "`git submodule update --init crates/backends/rlx-mlx-sys/vendor/mlx`.",
        detect: None,
    },
    KnownIssue {
        id: "RLX-BUILD-005",
        title: "ONNX Runtime prebuilt needs glibc 2.38+",
        severity: Severity::Error,
        symptom: "undefined reference to `__isoc23_strtol` (ort-sys)",
        cause: "The prebuilt libonnxruntime.a that ort-sys downloads targets glibc 2.38.",
        fix: "Build on a distro with glibc >= 2.38 (Ubuntu 24.04+, Debian trixie), or \
              disable the onnx feature.",
        detect: None,
    },
];

/// The issues that apply to `facts`, most severe first.
pub fn matching(facts: &HostFacts) -> Vec<&'static KnownIssue> {
    let mut v: Vec<&'static KnownIssue> = ISSUES
        .iter()
        .filter(|i| i.detect.is_some_and(|d| d(facts)))
        .collect();
    v.sort_by_key(|i| i.severity);
    v
}

pub fn by_id(id: &str) -> Option<&'static KnownIssue> {
    ISSUES.iter().find(|i| i.id == id)
}

/// `docs/known-issues.md`, generated from [`ISSUES`].
pub fn markdown() -> String {
    let mut s = String::from(
        "# Known issues\n\n\
         Generated from [`known_issues.rs`](../crates/core/rlx-runtime/src/doctor/known_issues.rs) \
         — do not edit by hand; run `just gen-known-issues`.\n\n\
         `cargo rlx doctor` detects the entries marked *detected* on your machine and \
         prints the fix. Build-time entries stop the build before `doctor` can run; \
         search this page for the error text.\n\n",
    );
    s.push_str("| Id | Severity | Title | |\n|----|----------|-------|---|\n");
    for i in ISSUES {
        s.push_str(&format!(
            "| [{id}](#{anchor}) | {sev} | {title} | {how} |\n",
            id = i.id,
            anchor = i.id.to_lowercase(),
            sev = i.severity.label(),
            title = i.title,
            how = if i.detect.is_some() {
                "detected"
            } else {
                "build time"
            },
        ));
    }
    for i in ISSUES {
        s.push_str(&format!(
            "\n## {id}\n\n**{title}** ({sev})\n\n- **Symptom:** `{symptom}`\n- **Cause:** {cause}\n- **Fix:** {fix}\n",
            id = i.id,
            title = i.title,
            sev = i.severity.label(),
            symptom = i.symptom,
            cause = i.cause,
            fix = i.fix,
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::super::facts::DeviceFact;
    use super::*;
    use std::path::Path;

    fn root(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rlx_ki_{name}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }
    fn touch(root: &Path, abs: &str, body: &str) {
        let p = root.join(abs.trim_start_matches('/'));
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, body).unwrap();
    }
    fn dev(device: Device, status: DeviceStatus, reason: &str) -> DeviceFact {
        DeviceFact {
            device,
            status,
            reason: reason.into(),
        }
    }
    fn facts(name: &str) -> HostFacts {
        HostFacts {
            root: root(name),
            ..Default::default()
        }
    }
    fn ids(f: &HostFacts) -> Vec<&'static str> {
        matching(f).iter().map(|i| i.id).collect()
    }

    /// One fake host per detected issue. Each must trigger its own id — the
    /// coverage test below fails if an entry with a `detect` has no case here.
    fn cases() -> Vec<(&'static str, HostFacts)> {
        let mut v = Vec::new();

        let mut f = facts("xdna001");
        f.versions = vec![("mlir_aie", (1, 5, 0))];
        v.push(("RLX-XDNA-001", f));

        for (id, reason) in [
            (
                "RLX-XDNA-002",
                "RLX_XDNA_SHIM is not set, and building it needs the XRT headers (xrt/xrt_device.h)",
            ),
            (
                "RLX-XDNA-003",
                "AMD XDNA NPU detected at /dev/accel/accel0 but no userspace runtime is available",
            ),
            (
                "RLX-XDNA-004",
                "/dev/accel/accel0 is not read/write for this user",
            ),
            ("RLX-XDNA-005", "RLIMIT_MEMLOCK is 8 MiB"),
            (
                "RLX-XDNA-006",
                "this is an XDNA 2 (Strix-class, AIE2P) NPU (8 AIE columns)",
            ),
            (
                "RLX-XDNA-007",
                "no mlir-aie toolchain found (searched PATH, ...)",
            ),
            (
                "RLX-XDNA-009",
                "mlir-aie at ~/old ships only aiecc.py (releases before 1.3 ...)",
            ),
            (
                "RLX-XDNA-010",
                "mlir-aie 1.4.4 pins Peano 22.0.0.2026092101, found 21.0.0.2026072001",
            ),
        ] {
            let mut f = facts(&id.to_lowercase());
            f.devices = vec![dev(Device::Xdna, DeviceStatus::Unavailable, reason)];
            v.push((id, f));
        }

        let mut f = facts("xdna008");
        touch(&f.root, "/home/u/xrt-root/usr/lib/libxrt_hip.so", "");
        f.env = vec![("LD_LIBRARY_PATH".into(), "/home/u/xrt-root/usr/lib".into())];
        f.devices = vec![dev(Device::Rocm, DeviceStatus::Ready, "")];
        v.push(("RLX-XDNA-008", f));

        let mut f = facts("rocm001");
        f.devices = vec![dev(Device::Rocm, DeviceStatus::Ready, "")];
        v.push(("RLX-ROCM-001", f));

        let mut f = facts("rocm002");
        touch(
            &f.root,
            "/sys/class/kfd/kfd/topology/nodes/1/properties",
            "simd_count 24\ngfx_target_version 110003\n",
        );
        f.devices = vec![dev(Device::Rocm, DeviceStatus::Unavailable, "")];
        v.push(("RLX-ROCM-002", f));

        let mut f = facts("rocm003");
        touch(&f.root, "/proc/sys/kernel/numa_balancing", "1\n");
        touch(&f.root, "/sys/devices/system/node/node1/meminfo", "");
        touch(
            &f.root,
            "/sys/class/kfd/kfd/topology/nodes/1/properties",
            "simd_count 24\nlocal_mem_size 0\ngfx_target_version 110003\n",
        );
        f.devices = vec![dev(Device::Rocm, DeviceStatus::Ready, "")];
        v.push(("RLX-ROCM-003", f));

        let mut f = facts("xdna011");
        f.versions = vec![("mlir_aie", (1, 3, 3))];
        v.push(("RLX-XDNA-011", f));

        let mut f = facts("cuda001");
        f.versions = vec![("cuda_driver", (12, 2, 0)), ("nvrtc", (12, 2, 0))];
        v.push(("RLX-CUDA-001", f));

        let mut f = facts("cuda002");
        f.versions = vec![("cuda_driver", (13, 2, 0))];
        v.push(("RLX-CUDA-002", f));

        let mut f = facts("cuda003");
        f.devices = vec![dev(Device::Cuda, DeviceStatus::Unavailable, "")];
        v.push(("RLX-CUDA-003", f));

        let mut f = facts("wgpu001");
        f.wgpu_host_routed = true;
        v.push(("RLX-WGPU-001", f));

        let mut f = facts("vulkan001");
        f.devices = vec![dev(Device::Vulkan, DeviceStatus::Unavailable, "")];
        v.push(("RLX-VULKAN-001", f));

        let mut f = facts("env001");
        f.env = vec![("RLX_DISPATCH_REPOT".into(), "1".into())]; // env-lint: ignore
        v.push(("RLX-ENV-001", f));

        let mut f = facts("check001");
        f.devices = vec![dev(Device::Gpu, DeviceStatus::Broken, "check graph failed")];
        v.push(("RLX-CHECK-001", f));
        v
    }

    #[test]
    fn each_fake_host_triggers_its_issue() {
        for (id, f) in cases() {
            assert!(
                ids(&f).contains(&id),
                "{id} not detected; got {:?}",
                ids(&f)
            );
        }
    }

    #[test]
    fn every_detected_issue_has_a_fake_host_case() {
        let covered: Vec<&str> = cases().iter().map(|(id, _)| *id).collect();
        for i in ISSUES.iter().filter(|i| i.detect.is_some()) {
            assert!(
                covered.contains(&i.id),
                "{} has a detect check but no test case",
                i.id
            );
        }
    }

    /// The healthy counterparts stay quiet: a working NPU, MIOpen with its
    /// database, the 780M already overridden, a Vulkan ICD, a known RLX_ var.
    #[test]
    fn healthy_hosts_stay_quiet() {
        let mut f = facts("healthy");
        touch(&f.root, "/opt/rocm/share/miopen/db/gfx1100.kdb", "");
        touch(&f.root, "/usr/share/vulkan/icd.d/radeon_icd.json", "{}");
        touch(&f.root, "/dev/nvidia0", "");
        touch(
            &f.root,
            "/sys/class/kfd/kfd/topology/nodes/1/properties",
            "gfx_target_version 110003\n",
        );
        f.env = vec![
            ("HSA_OVERRIDE_GFX_VERSION".into(), "11.0.0".into()),
            ("RLX_DISPATCH_REPORT".into(), "1".into()),
            ("LD_LIBRARY_PATH".into(), "/usr/lib".into()),
        ];
        f.versions = vec![
            ("cuda_driver", (13, 2, 0)),
            ("nvrtc", (13, 1, 0)),
            ("mlir_aie", (1, 4, 4)),
        ];
        f.devices = vec![
            dev(Device::Xdna, DeviceStatus::Ready, ""),
            dev(Device::Rocm, DeviceStatus::Ready, ""),
            dev(Device::Cuda, DeviceStatus::Ready, ""),
            dev(Device::Vulkan, DeviceStatus::Ready, ""),
        ];
        assert_eq!(ids(&f), Vec::<&str>::new());
    }

    #[test]
    fn ids_are_unique_and_well_formed() {
        let mut seen = std::collections::HashSet::new();
        for i in ISSUES {
            assert!(seen.insert(i.id), "duplicate id {}", i.id);
            let parts: Vec<&str> = i.id.split('-').collect();
            assert!(
                parts.len() == 3
                    && parts[0] == "RLX"
                    && parts[2].len() == 3
                    && parts[2].chars().all(|c| c.is_ascii_digit()),
                "malformed id {}",
                i.id
            );
        }
    }

    /// `docs/known-issues.md` is generated from the table.
    #[test]
    fn known_issues_doc_is_up_to_date() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/known-issues.md");
        let on_disk = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            on_disk == markdown(),
            "docs/known-issues.md is stale — run `just gen-known-issues`"
        );
    }
}
