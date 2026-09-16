// RLX — versatile ML compiler + runtime.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **Node hardware capabilities** — what each machine can contribute. Probed
//! locally (system queries + micro-benchmarks) and shipped between nodes as JSON
//! so a coordinator can plan placement without hard-coding the cluster.

use rlx_runtime::{Device, device_label, is_available, parse_device};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// A compute device present on a node, with its usable memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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
    /// Measured f32 matmul throughput on THIS device (GFLOP/s), 0 when the
    /// backend is not built in or the bench did not run.
    ///
    /// Placement needs this per device, not per node: ranking a machine by its
    /// CPU tells you nothing about the 4090 it will actually run the stage on.
    #[serde(default)]
    pub gflops: f64,
    /// True when the *node's own build* can actually target this device.
    ///
    /// The OS reporting a GPU and the binary being able to run on it are
    /// different facts: a build without the `metal` feature still sees an M4
    /// Pro, but every `Session::new(Metal)` on it panics. Recording that here
    /// keeps the distinction visible — the alternative, leaving `gflops` at
    /// 0.0, reads to a planner as "present but infinitely slow" and to a human
    /// as a benchmark failure, when the real answer is "rebuild with
    /// `--features metal`".
    #[serde(default = "default_true")]
    pub available: bool,
}

fn default_true() -> bool {
    true
}

impl DeviceInfo {
    fn new(device: Device, name: String, mem_bytes: u64, unified: bool) -> Self {
        Self {
            device: device_label(device).to_string(),
            name,
            mem_bytes,
            unified,
            gflops: 0.0,
            available: is_available(device),
        }
    }
    /// Parsed device kind, or `None` when this coordinator does not recognise
    /// the label.
    ///
    /// Worth distinguishing: nodes report whatever their own rlx build calls a
    /// device, so a cluster running mixed versions can send back a name this
    /// binary has never heard of. Collapsing that to `Cpu` — as [`Self::kind`]
    /// must, to keep its infallible signature — makes an unrecognised
    /// accelerator *invisible*: the planner skips it as though it were the host
    /// CPU and ignores its memory.
    pub fn parsed(&self) -> Option<Device> {
        parse_device(&self.device).ok()
    }

    /// Parsed device kind, falling back to CPU for an unrecognised label.
    ///
    /// Prefer [`Self::parsed`] where "unrecognised" and "CPU" mean different
    /// things — which is most places.
    pub fn kind(&self) -> Device {
        self.parsed().unwrap_or(Device::Cpu)
    }

    /// True when this coordinator understands the reported device.
    pub fn is_recognized(&self) -> bool {
        self.parsed().is_some()
    }

    /// True when a stage may be *placed* here: recognised by this coordinator
    /// and runnable by the node that reported it.
    pub fn is_usable(&self) -> bool {
        self.is_recognized() && self.available
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
            .filter(|d| d.is_usable() && d.kind() != Device::Cpu && d.mem_bytes > 0)
            .map(|d| d.mem_bytes)
            .max()
            .unwrap_or(self.ram_total)
    }

    /// Measured throughput of `dev` specifically, falling back to the node's
    /// headline figure when that device was not benched.
    ///
    /// Costing a stage by [`Self::gflops`] — the node's *fastest* device — is
    /// only right when the stage runs there, and it need not: the device is
    /// picked for its memory ceiling (or pinned by hand in the config). On an
    /// M4 Pro the CPU's AMX benches ~1100 GFLOP/s against Metal's ~540, so a
    /// node placed on Metal and costed on the headline is priced at twice the
    /// throughput it will deliver.
    pub fn gflops_for(&self, dev: Device) -> f64 {
        self.devices
            .iter()
            .find(|d| d.parsed() == Some(dev) && d.gflops > 0.0)
            .map(|d| d.gflops)
            .unwrap_or(self.gflops)
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

/// Time a real f32 matmul on `dev` through rlx itself → GFLOP/s.
///
/// The point is to rank *accelerators*: the host-side scalar loop in
/// [`bench_gflops`] says a 4090 node and a laptop are the same speed, which is
/// exactly backwards for placement.
///
/// Wrapped in `catch_unwind` deliberately. Backend availability is not reliably
/// reportable — a backend can be compiled in, claim support, and still abort on
/// a machine without the driver — and a hardware probe must degrade to "unknown"
/// rather than take the coordinator down. `None` means "could not measure",
/// which the planner treats as no information rather than as zero speed.
fn bench_device_gflops(dev: Device) -> Option<f64> {
    use rlx_ir::{DType, Graph, Shape};
    // Asking first is not just tidiness: compiling for a device the build lacks
    // panics, and while `catch_unwind` below contains it, the default hook still
    // prints a backtrace-shaped line per device per node. A clean probe that
    // prints three panics looks like a broken probe.
    if !is_available(dev) {
        return None;
    }
    // Sized and timed to survive being a *ranking* input. The previous
    // 4 iterations of 256^3 was 0.34 ms of work on an M4 Pro, so what it
    // actually measured was timer granularity, thread-pool spin-up and clock
    // ramp: the same machine probed three times reported 276, 331 and 493
    // GFLOP/s, and the planner moved layers around on the strength of it.
    // Shape matters as much as duration. A square 512^3 GEMM is compute-bound
    // and reports what the AMX/tensor units can do; an autoregressive decode
    // step is a batch-1 GEMV against a full weight matrix, which is bound by
    // how fast the machine can stream those weights. Benching the square case
    // and then planning decode stages with it overstates throughput by roughly
    // an order of magnitude, and overstates it MOST on the nodes with the
    // widest gap between arithmetic and bandwidth. Bench what we are planning.
    let n = 4096usize;
    let rows = 1usize;
    /// Each timing round must run at least this long to swamp the fixed costs.
    const ROUND_SECS: f64 = 0.03;
    /// Best-of, not mean: interference from other work can only ever make a
    /// round look slower, so the fastest round is the closest to the truth.
    const ROUNDS: usize = 3;

    let run_once = || -> Option<f64> {
        let mut g = Graph::new("caps_bench");
        let a = g.input("a", Shape::new(&[rows, n], DType::F32));
        let b = g.param("b", Shape::new(&[n, n], DType::F32));
        let c = g.matmul(a, b, Shape::new(&[rows, n], DType::F32));
        g.set_outputs(vec![c]);
        let mut compiled = rlx_runtime::Session::new(dev).compile(g);
        compiled.set_param("b", &vec![0.5f32; n * n]);
        let x = vec![0.25f32; rows * n];
        // Warm-up: compile, upload, JIT, and let the clocks come up.
        let t = Instant::now();
        let _ = compiled.run(&[("a", x.as_slice())]);
        let first = t.elapsed().as_secs_f64().max(1e-6);
        // Enough iterations that one round lasts ROUND_SECS at the warm-up rate.
        let iters = ((ROUND_SECS / first).ceil() as usize).clamp(2, 4096);
        let flop = 2.0 * (rows * n * n) as f64 * iters as f64;
        let mut best = 0.0f64;
        for _ in 0..ROUNDS {
            let t = Instant::now();
            for _ in 0..iters {
                let _ = compiled.run(&[("a", x.as_slice())]);
            }
            let secs = t.elapsed().as_secs_f64();
            if secs > 0.0 {
                best = best.max(flop / secs / 1e9);
            }
        }
        (best > 0.0).then_some(best)
    };
    // A backend can still fail past the availability check (no device present,
    // driver refused, unsupported op). That is a benign "no measurement", so
    // keep the unwind but drop the hook's stderr spew for its duration.
    // The hook is process-global, so two benches racing here would restore each
    // other's hook and leak the silent one. One at a time.
    static HOOK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = HOOK.lock().unwrap_or_else(|e| e.into_inner());
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run_once))
        .ok()
        .flatten();
    std::panic::set_hook(hook);
    out
}

/// Ask the kernel not to keep this bench's own reads in the page cache, so a
/// probe does not poison the next one.
///
/// Advisory on both platforms; note it does NOT evict pages that are already
/// resident, which is why the caller must also pick a cold offset.
#[cfg(unix)]
fn no_cache_hint(f: &std::fs::File, off: u64, len: u64) {
    use std::os::fd::AsRawFd;
    let fd = f.as_raw_fd();
    // SAFETY: `fd` is owned by `f` and outlives the call; both are advisory
    // hints whose failure we deliberately ignore.
    unsafe {
        #[cfg(target_vendor = "apple")]
        {
            let _ = (off, len);
            libc::fcntl(fd, libc::F_NOCACHE, 1);
        }
        #[cfg(target_os = "linux")]
        {
            libc::posix_fadvise(fd, off as i64, len as i64, libc::POSIX_FADV_DONTNEED);
        }
        #[cfg(not(any(target_vendor = "apple", target_os = "linux")))]
        {
            let _ = (fd, off, len);
        }
    }
}

#[cfg(not(unix))]
fn no_cache_hint(_f: &std::fs::File, _off: u64, _len: u64) {}

/// Fraction of the pages in `[off, off+len)` already in the page cache, or
/// `None` when the question cannot be answered.
///
/// This is the difference between measuring a disk and measuring RAM. Timing a
/// read of cached pages on this M4 Pro returns ~20 GB/s; the same range read
/// cold off the actual device returns ~0.9 GB/s. Twenty-fold, and in the
/// dangerous direction: `stage_secs` divides the expert bytes each token touches
/// by this number, so a cached probe says paging 86 GB of experts off disk is
/// nearly free and the planner cheerfully builds a stage that will be an order
/// of magnitude slower than predicted.
///
/// A varying offset alone does not save us — it only helps while the file is
/// bigger than free RAM, and it never helps on a re-probe of a small one.
#[cfg(unix)]
fn resident_fraction(f: &std::fs::File, off: u64, len: usize) -> Option<f64> {
    use std::os::fd::AsRawFd;
    // SAFETY: sysconf(_SC_PAGESIZE) takes no pointers and cannot fail here.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page <= 0 {
        return None;
    }
    let page = page as u64;
    // mmap needs a page-aligned offset; widen the window to cover the request.
    let base = off - (off % page);
    let span = (len as u64 + (off - base)) as usize;
    let pages = span.div_ceil(page as usize);

    // SAFETY: a read-only shared mapping of a file we hold open. `addr` is
    // checked against MAP_FAILED before use and unmapped on every path.
    unsafe {
        let addr = libc::mmap(
            std::ptr::null_mut(),
            span,
            libc::PROT_READ,
            libc::MAP_SHARED,
            f.as_raw_fd(),
            base as libc::off_t,
        );
        if addr == libc::MAP_FAILED {
            return None;
        }
        let mut vec = vec![0u8; pages];
        let rc = libc::mincore(addr, span, vec.as_mut_ptr().cast());
        libc::munmap(addr, span);
        if rc != 0 {
            return None;
        }
        // Bit 0 is "resident" on both Linux and the BSDs.
        let hot = vec.iter().filter(|b| *b & 1 != 0).count();
        Some(hot as f64 / pages.max(1) as f64)
    }
}

#[cfg(not(unix))]
fn resident_fraction(_f: &std::fs::File, _off: u64, _len: usize) -> Option<f64> {
    None
}

/// Sequential read throughput (MB/s) at the checkpoint location.
///
/// Reads a slice of the LARGEST file already in `dir` — for a cluster node that
/// is the checkpoint itself, i.e. the exact medium the routed experts will page
/// from. Falls back to a write+fsync probe when the directory is empty, which
/// still separates NVMe from SATA from spinning rust even if the absolute number
/// is a write figure.
///
/// This used to be hardcoded to 0.0, which silently made every node look equally
/// slow to any IO-aware policy.
fn bench_io_mbps(dir: &str) -> f64 {
    const CHUNK: usize = 64 << 20;
    // How many offsets to try before concluding the file is simply all cached.
    const TRIES: usize = 8;
    // Above this, the window is cache-served and timing it measures RAM.
    const COLD: f64 = 0.10;

    if let Some((path, len)) = largest_file(dir)
        && len > (8 << 20)
        && let Ok(mut f) = std::fs::File::open(&path)
    {
        use std::io::{Read, Seek, SeekFrom};
        let want = CHUNK.min(len as usize);
        let span = len as usize - want;
        // Look for a window the page cache does not already hold. Without this
        // the number is RAM bandwidth — see `resident_fraction`.
        let mut chosen = None;
        for i in 0..TRIES {
            let off = if span > 0 {
                // Vary per attempt and per run; no RNG dependency needed.
                (nanos() as usize).wrapping_mul(i * 2 + 1) % span
            } else {
                0
            };
            match resident_fraction(&f, off as u64, want) {
                // Cannot tell (no mincore, mmap refused): the historical
                // behaviour, one read at a varying offset, is the best we have.
                None => {
                    chosen = Some(off);
                    break;
                }
                Some(r) if r <= COLD => {
                    chosen = Some(off);
                    break;
                }
                Some(_) => {
                    if span == 0 {
                        break; // Only one window exists and it is hot.
                    }
                }
            }
        }
        if let Some(off) = chosen {
            no_cache_hint(&f, off as u64, want as u64);
            if f.seek(SeekFrom::Start(off as u64)).is_ok() {
                let mut buf = vec![0u8; want];
                let t = Instant::now();
                if f.read_exact(&mut buf).is_ok() {
                    let secs = t.elapsed().as_secs_f64();
                    std::hint::black_box(&buf);
                    if secs > 0.0 {
                        return want as f64 / 1e6 / secs;
                    }
                }
            }
        }
        // Every window sampled was resident — typical when the checkpoint is
        // smaller than free RAM. Reporting that read would claim ~20 GB/s of
        // "disk". Fall through to the write probe, which at least has to reach
        // the device.
    }
    write_probe_mbps(dir)
}

fn nanos() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(1)
}

/// Largest regular file directly in `dir` (not recursive).
fn largest_file(dir: &str) -> Option<(std::path::PathBuf, u64)> {
    let rd = std::fs::read_dir(dir).ok()?;
    rd.flatten()
        .filter_map(|e| {
            let md = e.metadata().ok()?;
            md.is_file().then(|| (e.path(), md.len()))
        })
        .max_by_key(|(_, l)| *l)
}

/// Write + fsync a small file and time it — the fallback when there is nothing
/// to read. Cleans up after itself.
fn write_probe_mbps(dir: &str) -> f64 {
    use std::io::Write;
    const N: usize = 16 << 20;
    let path = std::path::Path::new(dir).join(format!(".rlx-io-probe-{}", nanos()));
    let buf = vec![0xA5u8; N];
    let t = Instant::now();
    let ok = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&path)?;
        f.write_all(&buf)?;
        f.sync_all()
    })()
    .is_ok();
    let secs = t.elapsed().as_secs_f64();
    let _ = std::fs::remove_file(&path);
    if ok && secs > 0.0 {
        N as f64 / 1e6 / secs
    } else {
        0.0
    }
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
    let mut devices = detect_devices(&os);
    if bench {
        for d in devices.iter_mut() {
            d.gflops = bench_device_gflops(d.kind()).unwrap_or(0.0);
        }
    }
    // The node's headline throughput is its FASTEST device, not its CPU — that
    // is the one a stage will actually run on. Fall back to the host loop only
    // when no device could be measured.
    let gflops = if bench {
        devices
            .iter()
            .map(|d| d.gflops)
            .fold(0.0f64, f64::max)
            .max(0.0)
    } else {
        0.0
    };
    let gflops = if bench && gflops <= 0.0 {
        bench_gflops()
    } else {
        gflops
    };
    NodeCaps {
        addr: addr.to_string(),
        cores: std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        ram_total,
        ram_avail,
        disk_free: disk_free(ckpt_dir),
        devices,
        gflops,
        io_mbps: if bench { bench_io_mbps(ckpt_dir) } else { 0.0 },
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

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str, bytes: usize) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("rlx-caps-{name}"));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("probe.bin");
        std::fs::write(&p, vec![7u8; bytes]).unwrap();
        p
    }

    /// `mincore` has to actually answer, or the cold-window search silently
    /// degrades to the old behaviour of timing the page cache.
    #[cfg(unix)]
    #[test]
    fn residency_is_observable_and_tracks_reads() {
        let p = scratch("residency", 4 << 20);
        let f = std::fs::File::open(&p).unwrap();
        // Just written, so the pages are dirty in cache: definitely resident.
        let hot = resident_fraction(&f, 0, 4 << 20)
            .expect("mincore must answer on a plain file-backed mapping");
        assert!(
            hot > 0.5,
            "a file this process just wrote reads as {hot:.2} resident — the \
             residency probe is not seeing the page cache, so the disk bench \
             cannot tell RAM from storage"
        );
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    /// A checkpoint smaller than RAM is entirely cached, so every window is hot
    /// and the read timing would report RAM bandwidth. The bench must fall back
    /// rather than return ~20 GB/s of imaginary disk.
    #[test]
    fn a_fully_cached_checkpoint_does_not_report_ram_bandwidth() {
        let p = scratch("cached", 32 << 20);
        let dir = p.parent().unwrap().to_str().unwrap().to_string();
        // Pull it all into cache first, the way loading a model would.
        let _ = std::fs::read(&p).unwrap();
        let mbps = bench_io_mbps(&dir);
        assert!(mbps > 0.0 && mbps.is_finite(), "{mbps}");
        assert!(
            mbps < 10_000.0,
            "{mbps:.0} MB/s is memory bandwidth, not a disk — the planner would \
             price expert paging at roughly nothing"
        );
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }
}
