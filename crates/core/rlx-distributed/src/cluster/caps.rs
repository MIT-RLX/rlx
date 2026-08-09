// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Node hardware capabilities** — what each machine can contribute. Probed
//! locally (system queries + micro-benchmarks) and shipped between nodes as JSON
//! so a coordinator can plan placement without hard-coding the cluster.

use rlx_runtime::{Device, device_label, parse_device};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A compute device present on a node, with its usable memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DeviceInfo {
    /// RLX device label (`cpu`, `metal`, `cuda`, `ane`/NPU, `vulkan`, …). Stored
    /// as its canonical string so [`NodeCaps`] stays plainly (de)serializable.
    pub device: String,
    /// Human name (e.g. "Apple M4 Pro", "NVIDIA RTX 3080 Ti").
    pub name: String,
    /// Dedicated device memory in bytes (0 / host-RAM for unified/iGPU).
    pub mem_bytes: u64,
    /// True when the device shares host RAM (Apple unified, iGPU) — no separate
    /// VRAM ceiling, but it competes with the CPU stage for the same pool.
    pub unified: bool,
}

impl DeviceInfo {
    fn new(device: Device, name: String, mem_bytes: u64, unified: bool) -> Self {
        Self {
            device: device_label(device).to_string(),
            name,
            mem_bytes,
            unified,
        }
    }
    /// Parsed device kind.
    pub fn kind(&self) -> Device {
        parse_device(&self.device).unwrap_or(Device::Cpu)
    }
}

/// A node's measured hardware profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeCaps {
    /// Address the coordinator reaches this node's worker at (`host:port`).
    pub addr: String,
    /// OS string (`macos`, `linux`).
    pub os: String,
    /// Logical CPU cores.
    pub cores: usize,
    /// Total / currently-available host RAM (bytes).
    pub ram_total: u64,
    pub ram_avail: u64,
    /// Free disk at the checkpoint location (bytes).
    pub disk_free: u64,
    /// Compute devices, CPU first.
    pub devices: Vec<DeviceInfo>,
    /// Rough sustained matmul throughput (GFLOP/s, f32) from a micro-bench.
    pub gflops: f64,
    /// Rough disk read throughput (MB/s) from a micro-bench (0 if not measured).
    pub io_mbps: f64,
}

impl NodeCaps {
    /// Best (non-CPU) accelerator memory ceiling, else host RAM — the largest a
    /// stage can occupy on this node's device. A GPU that reports a positive
    /// `mem_bytes` (discrete VRAM, or an APU's GTT / GPU-addressable system RAM)
    /// caps the stage even when it is "unified": an amdgpu APU shares system RAM
    /// but the kernel bounds GPU access at GTT, so we must honor it. Devices with
    /// no meaningful ceiling (Apple Metal, `mem_bytes == 0`) fall through to RAM.
    pub fn accel_mem(&self) -> u64 {
        self.devices
            .iter()
            .filter(|d| d.kind() != Device::Cpu && d.mem_bytes > 0)
            .map(|d| d.mem_bytes)
            .max()
            .unwrap_or(self.ram_total)
    }

    /// One-line summary for logs / the monitor table.
    pub fn summary(&self) -> String {
        let devs: Vec<String> = self
            .devices
            .iter()
            .map(|d| device_label(d.kind()).to_string())
            .collect();
        format!(
            "{} | {} cores | {:.0}/{:.0} GB RAM | {:.0} GFLOP/s | {}",
            self.os,
            self.cores,
            self.ram_avail as f64 / 1e9,
            self.ram_total as f64 / 1e9,
            self.gflops,
            devs.join("+"),
        )
    }
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(cmd)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
}

/// Quick f32 matmul micro-bench → GFLOP/s (single-thread; a throughput proxy).
fn bench_gflops() -> f64 {
    let n = 384usize;
    let a = vec![1.0000117f32; n * n];
    let b = vec![0.9999931f32; n * n];
    let mut c = vec![0f32; n * n];
    let t = Instant::now();
    for i in 0..n {
        for k in 0..n {
            let av = a[i * n + k];
            let brow = &b[k * n..k * n + n];
            let crow = &mut c[i * n..i * n + n];
            for j in 0..n {
                crow[j] += av * brow[j];
            }
        }
    }
    std::hint::black_box(&c);
    let secs = t.elapsed().as_secs_f64();
    (2.0 * (n * n * n) as f64) / secs / 1e9
}

fn detect_devices(os: &str) -> Vec<DeviceInfo> {
    let mut out = vec![DeviceInfo::new(Device::Cpu, cpu_name(os), 0, false)];
    if os == "macos" {
        // Apple Silicon: unified-memory Metal/MLX GPU + Neural Engine.
        let name = cpu_name(os);
        out.push(DeviceInfo::new(Device::Metal, name.clone(), 0, true));
        out.push(DeviceInfo::new(
            Device::Ane,
            format!("{name} Neural Engine"),
            0,
            true,
        ));
    } else {
        // NVIDIA via nvidia-smi.
        if let Some(s) = run(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
        ) {
            for line in s.lines() {
                let mut it = line.split(',');
                let name = it.next().unwrap_or("CUDA GPU").trim().to_string();
                let mem_mib: u64 = it.next().and_then(|m| m.trim().parse().ok()).unwrap_or(0);
                out.push(DeviceInfo::new(
                    Device::Cuda,
                    name,
                    mem_mib * 1024 * 1024,
                    false,
                ));
            }
        }
        // AMD/Vulkan iGPU (shared RAM). Report GTT — the GPU-addressable slice of
        // system RAM (amdgpu sysfs) — as the memory ceiling. A Vulkan arena OOMs
        // if a stage exceeds it, and unlike CUDA managed memory it cannot page
        // past GTT, so the planner must treat it as a hard cap.
        let amd_gtt = amdgpu_gtt_total();
        if run("rocminfo", &[]).is_some() {
            out.push(DeviceInfo::new(
                Device::Rocm,
                "AMD ROCm".into(),
                amd_gtt,
                true,
            ));
        } else if run("vulkaninfo", &["--summary"])
            .map(|s| s.contains("GPU"))
            .unwrap_or(false)
            || amd_gtt > 0
        {
            out.push(DeviceInfo::new(
                Device::Vulkan,
                "Vulkan GPU".into(),
                amd_gtt,
                true,
            ));
        }
    }
    out
}

/// AMD APU GPU-addressable system memory (GTT) in bytes, from amdgpu sysfs.
/// GTT is the real ceiling for a Vulkan/ROCm arena on an iGPU (VRAM is a tiny
/// BAR carve-out). Returns 0 when not an amdgpu system.
fn amdgpu_gtt_total() -> u64 {
    let Ok(rd) = std::fs::read_dir("/sys/class/drm") else {
        return 0;
    };
    for ent in rd.flatten() {
        let name = ent.file_name();
        let name = name.to_string_lossy();
        // Match `card0`, `card1`, … (skip connector nodes like `card0-eDP-1`).
        if !name.starts_with("card") || name.contains('-') {
            continue;
        }
        let p = ent.path().join("device/mem_info_gtt_total");
        if let Ok(s) = std::fs::read_to_string(&p)
            && let Ok(v) = s.trim().parse::<u64>()
            && v > 0
        {
            return v;
        }
    }
    0
}

fn cpu_name(os: &str) -> String {
    if os == "macos" {
        run("sysctl", &["-n", "machdep.cpu.brand_string"])
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "CPU".into())
    } else {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("model name"))
                    .map(|l| l.split(':').nth(1).unwrap_or("CPU").trim().to_string())
            })
            .unwrap_or_else(|| "CPU".into())
    }
}

fn ram(os: &str) -> (u64, u64) {
    if os == "macos" {
        let total: u64 = run("sysctl", &["-n", "hw.memsize"])
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);
        // Available ≈ free + inactive pages × page size (vm_stat).
        let page: u64 = run("sysctl", &["-n", "hw.pagesize"])
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(16384);
        let vm = run("vm_stat", &[]).unwrap_or_default();
        let grab = |key: &str| -> u64 {
            vm.lines()
                .find(|l| l.contains(key))
                .and_then(|l| l.rsplit(' ').next())
                .map(|n| n.trim_end_matches('.').parse().unwrap_or(0))
                .unwrap_or(0)
        };
        let avail =
            (grab("Pages free:") + grab("Pages inactive:") + grab("Pages purgeable:")) * page;
        (total, avail.min(total))
    } else {
        let mi = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let kb = |key: &str| -> u64 {
            mi.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|n| n.parse::<u64>().ok())
                .unwrap_or(0)
                * 1024
        };
        (kb("MemTotal:"), kb("MemAvailable:"))
    }
}

fn disk_free(dir: &str) -> u64 {
    // `df -k <dir>` → available KiB in the 4th column of the data row.
    run("df", &["-k", dir])
        .and_then(|s| s.lines().nth(1).map(String::from))
        .and_then(|row| {
            row.split_whitespace()
                .nth(3)
                .and_then(|k| k.parse::<u64>().ok())
        })
        .map(|k| k * 1024)
        .unwrap_or(0)
}

/// Probe THIS machine. `addr` is how the coordinator will reach its worker;
/// `ckpt_dir` sizes the disk check; `bench` runs the (short) FLOP micro-bench.
pub fn probe_local(addr: &str, ckpt_dir: &str, bench: bool) -> NodeCaps {
    let os = std::env::consts::OS.to_string();
    let (ram_total, ram_avail) = ram(&os);
    NodeCaps {
        addr: addr.to_string(),
        cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ram_total,
        ram_avail,
        disk_free: disk_free(ckpt_dir),
        devices: detect_devices(&os),
        gflops: if bench { bench_gflops() } else { 0.0 },
        io_mbps: 0.0,
        os,
    }
}

/// Probe a REMOTE node over SSH by running `<remote_bin> --probe --addr <addr>
/// --ckpt <dir>` there and parsing its JSON. The same worker binary self-reports.
pub fn probe_remote(
    ssh_host: &str,
    remote_bin: &str,
    addr: &str,
    ckpt_dir: &str,
) -> anyhow::Result<NodeCaps> {
    let cmd = format!("{remote_bin} --probe --addr {addr} --ckpt {ckpt_dir}");
    let out = std::process::Command::new("ssh")
        .arg(ssh_host)
        .arg(&cmd)
        .output()?;
    if !out.status.success() {
        anyhow::bail!(
            "probe {ssh_host} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let json = text
        .lines()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or("");
    serde_json::from_str(json)
        .map_err(|e| anyhow::anyhow!("parse caps from {ssh_host}: {e}: {json}"))
}
