// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Measured device placement for concurrent jobs.
//!
//! The cost model ([`crate::fastest_device_for`]) predicts; [`measured_device`]
//! checks: it times a few real steps of the caller's own workload on the cost
//! model's GPU pick and on the CPU, keeps the GPU only at a 1.5× margin, and
//! makes that decision once per key for the process (optionally remembered in
//! `RLX_PLACEMENT_CACHE` across runs). [`gpu_slot`] lets concurrent jobs share
//! a GPU, optionally capped (`RLX_GPU_JOBS`), and gives a probe the GPU to
//! itself so its timing is not another job's queueing; it also records which
//! devices each thread compiled for ([`take_devices_used`]).
//!
//! [`crate::device_bench`] ranks devices by benchmarking a compiled graph;
//! this module decides for a whole workload from timings the caller takes.
//! Moved from moabb's backend module, where it was written for exgb's
//! concurrent training folds.

use crate::Device;

/// How many training/inference jobs may use a GPU at once; `None` = no limit
/// (the default).
///
/// Jobs that run concurrently (e.g. exgb's fold pool, up to every core) can
/// oversubscribe a shared GPU. On this Mac it was not measurably: five transformer
/// decoders × 2 folds took 134 / 97 / 90 s at 2 / 4 / 14 jobs (one ordering
/// had shown 152 / 276 / 658 s — a load spike, the machine's load average
/// swung 45–70). So the cap is opt-in: `RLX_GPU_JOBS=n`, e.g. on a GPU shared
/// with other users. Scores are identical at any setting. CPU work is never
/// limited by it.
pub fn gpu_jobs() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        rlx_ir::env::var("RLX_GPU_JOBS")
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
    })
}

struct GpuSlots {
    /// `(running, next ticket, ticket now allowed in)` — first come, first served.
    state: std::sync::Mutex<(usize, u64, u64)>,
    freed: std::sync::Condvar,
}

static GPU_SLOTS: GpuSlots = GpuSlots {
    state: std::sync::Mutex::new((0, 0, 0)),
    freed: std::sync::Condvar::new(),
};

thread_local! {
    /// Devices this thread compiled graphs for since [`take_devices_used`] —
    /// what a benchmark cell reports as where it ran.
    static DEVICES_USED: std::cell::RefCell<std::collections::BTreeSet<String>> = const { std::cell::RefCell::new(std::collections::BTreeSet::new()) };
    /// The device [`measured_device`] last chose on this thread.
    static LAST_PLACEMENT: std::cell::Cell<Option<Device>> = const { std::cell::Cell::new(None) };
    /// Slots this thread holds: nested acquisitions (fit → its stats pass) do
    /// not wait on themselves.
    static HELD: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Every GPU job holds this shared; a placement probe takes it exclusively,
/// so its timings are not another job's queueing.
static GPU_USE: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// A held GPU slot; released on drop. Not `Send`: it belongs to its thread.
pub struct GpuSlot {
    counted: bool,
    _shared: Option<std::sync::RwLockReadGuard<'static, ()>>,
    _not_send: std::marker::PhantomData<*const ()>,
}

/// Take a GPU slot for work on `device` (none for the CPU, or when
/// [`gpu_jobs`] is unset), waiting while that many jobs hold one.
pub fn gpu_slot(device: Device) -> GpuSlot {
    DEVICES_USED.with(|d| d.borrow_mut().insert(format!("{device:?}").to_lowercase()));
    if matches!(device, Device::Cpu) {
        return GpuSlot {
            counted: false,
            _shared: None,
            _not_send: std::marker::PhantomData,
        };
    }
    if HELD.with(|h| h.get()) > 0 {
        HELD.with(|h| h.set(h.get() + 1));
        return GpuSlot {
            counted: true,
            _shared: None,
            _not_send: std::marker::PhantomData,
        };
    }
    let shared = Some(GPU_USE.read().unwrap_or_else(|e| e.into_inner()));
    let Some(cap) = gpu_jobs() else {
        HELD.with(|h| h.set(1));
        return GpuSlot {
            counted: true,
            _shared: shared,
            _not_send: std::marker::PhantomData,
        };
    };
    let mut st = GPU_SLOTS.state.lock().unwrap_or_else(|e| e.into_inner());
    let ticket = st.1;
    st.1 += 1;
    while st.0 >= cap || st.2 != ticket {
        st = GPU_SLOTS.freed.wait(st).unwrap_or_else(|e| e.into_inner());
    }
    st.0 += 1;
    st.2 += 1;
    drop(st);
    GPU_SLOTS.freed.notify_all();
    HELD.with(|h| h.set(1));
    GpuSlot {
        counted: true,
        _shared: shared,
        _not_send: std::marker::PhantomData,
    }
}

impl Drop for GpuSlot {
    fn drop(&mut self) {
        if !self.counted {
            return;
        }
        let left = HELD.with(|h| {
            let v = h.get().saturating_sub(1);
            h.set(v);
            v
        });
        if left == 0 && gpu_jobs().is_some() {
            GPU_SLOTS.state.lock().unwrap_or_else(|e| e.into_inner()).0 -= 1;
            GPU_SLOTS.freed.notify_all();
        }
    }
}

/// The device to train a graph on, **measured**: `candidate` is the cost
/// model's pick; when it is a GPU, `time_step(dev)` times a few real training
/// steps on it and on the CPU, and the GPU is kept only when it is at least
/// 1.5× faster. The choice is remembered per `key` (model + geometry) for the
/// process, so folds of one model probe once and agree.
///
/// Why: the cost model put EEGConformer's training graph on Metal, where 64
/// steps took 20–99 s against 1.7 s on the CPU, while conv nets ran several
/// times faster there. A static list would be wrong on other machines.
/// `RLX_FORCE_DEVICE` (or `RLX_FORCE_CPU`) skips the probe; `RLX_PLACEMENT=static`
/// keeps the cost model's pick. The margin keeps borderline graphs on the CPU,
/// where results are deterministic.
pub fn measured_device(
    key: &str,
    candidate: Device,
    time_step: impl FnMut(Device) -> Option<f64>,
) -> Device {
    let d = measured_device_inner(key, candidate, time_step);
    LAST_PLACEMENT.with(|l| l.set(Some(d)));
    d
}

/// The device the last [`measured_device`] call on this thread chose, cleared.
/// A model trained on it runs its inference graphs there too, so a cell does
/// not train on one device and predict on another.
pub fn take_last_placement() -> Option<Device> {
    LAST_PLACEMENT.with(|l| l.take())
}

fn measured_device_inner(
    key: &str,
    candidate: Device,
    mut time_step: impl FnMut(Device) -> Option<f64>,
) -> Device {
    if rlx_ir::env::var_os("RLX_FORCE_DEVICE").is_some()
        || rlx_ir::env::var_os("RLX_FORCE_CPU").is_some()
        || rlx_ir::env::var("RLX_PLACEMENT").as_deref() == Some("static")
    {
        return candidate;
    }
    type Choices = std::sync::Mutex<
        std::collections::HashMap<String, std::sync::Arc<std::sync::OnceLock<Device>>>,
    >;
    static CHOSEN: std::sync::OnceLock<Choices> = std::sync::OnceLock::new();
    let cell = CHOSEN
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(key.to_string())
        .or_default()
        .clone();
    // Decided once per key, whatever the cost model says on later calls: its
    // pick is not stable (rlx-metal re-measures its calibration when the
    // cache file is missing or caught mid-write), so a borderline graph could
    // otherwise train on the CPU in one fold and the GPU in the next — the
    // rare serial-vs-concurrent mismatch in exgb's determinism test.
    *cell.get_or_init(|| {
        if let Some(d) = placement_cache_get(key) {
            return d;
        }
        // the cost model's CPU pick is final (nothing to time against)
        if matches!(candidate, Device::Cpu) {
            return Device::Cpu;
        }
        // a thread holding a GPU slot cannot take the GPU exclusively: keep
        // the candidate rather than deadlock
        if HELD.with(|h| h.get()) > 0 {
            return candidate;
        }
        // the GPU to itself while timing (CPU work elsewhere still competes,
        // which can only make the CPU look slower)
        let excl = GPU_USE.write().unwrap_or_else(|e| e.into_inner());
        let gpu = time_step(candidate);
        let cpu = time_step(Device::Cpu);
        drop(excl);
        let pick = match (gpu, cpu) {
            (Some(g), Some(c)) if g * 1.5 < c => candidate,
            (Some(_), None) => candidate,
            _ => Device::Cpu,
        };
        if rlx_ir::env::var_os("RLX_LOG_DEVICE").is_some() {
            eprintln!(
                "[rlx] placement {key}: {candidate:?} {gpu:?}s/step vs Cpu {cpu:?}s/step → {pick:?}"
            );
        }
        placement_cache_put(key, pick, gpu, cpu);
        pick
    })
}

/// `RLX_PLACEMENT_CACHE` (a file; unset or `off` = decisions stay in memory):
/// one `key<TAB>device<TAB>gpu s/step<TAB>cpu s/step` line per decision, so
/// later runs place each model the same way without probing. Delete it to
/// re-measure (e.g. after a backend change).
///
/// There is no default file here: a library caller or a test (a test suite
/// probes under whatever else is using the GPU) must not write decisions that
/// every later run would trust. An application opts in (exgb sets
/// `~/.cache/exgb/placement.tsv`).
fn placement_cache_path() -> Option<std::path::PathBuf> {
    match rlx_ir::env::var("RLX_PLACEMENT_CACHE") {
        Some(v) if !v.is_empty() && v != "off" => Some(v.into()),
        _ => None,
    }
}

fn placement_cache_get(key: &str) -> Option<Device> {
    let text = std::fs::read_to_string(placement_cache_path()?).ok()?;
    text.lines().rev().find_map(|l| {
        let mut f = l.split('\t');
        (f.next()? == key)
            .then(|| f.next().and_then(|d| crate::parse_device(d).ok()))
            .flatten()
    })
}

fn placement_cache_put(key: &str, pick: Device, gpu: Option<f64>, cpu: Option<f64>) {
    use std::io::Write;
    let Some(path) = placement_cache_path() else {
        return;
    };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    static WRITE: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _g = WRITE.lock().unwrap_or_else(|e| e.into_inner());
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        let name = format!("{pick:?}").to_lowercase();
        let _ = f.write_all(
            format!(
                "{key}\t{name}\t{:.6}\t{:.6}\n",
                gpu.unwrap_or(f64::NAN),
                cpu.unwrap_or(f64::NAN)
            )
            .as_bytes(),
        );
    }
}

/// The devices this thread compiled training/inference graphs for since the
/// last call (every compile site takes a [`gpu_slot`], CPU included), and
/// clear them. A benchmark cell reports these as where it ran.
pub fn take_devices_used() -> std::collections::BTreeSet<String> {
    DEVICES_USED.with(|d| std::mem::take(&mut *d.borrow_mut()))
}

#[cfg(test)]
mod gpu_slot_tests {
    use super::*;

    /// The GPU is kept only at a 1.5× margin; a key is probed once per process
    /// and its decision is written to, then read back from, the cache file.
    #[test]
    fn measured_placement_needs_a_margin_and_remembers() {
        let path = std::env::temp_dir().join(format!("rlx_placement_{}.tsv", std::process::id()));
        let _ = std::fs::remove_file(&path);
        // SAFETY: test-only; this test is the only reader of the variable while it runs.
        unsafe { std::env::set_var("RLX_PLACEMENT_CACHE", &path) };
        rlx_ir::env::reload();
        if rlx_ir::env::var_os("RLX_FORCE_DEVICE").is_some()
            || rlx_ir::env::var_os("RLX_FORCE_CPU").is_some()
        {
            return;
        }
        let fake = |gpu: f64, cpu: f64| {
            move |d: Device| Some(if matches!(d, Device::Cpu) { cpu } else { gpu })
        };
        assert_eq!(
            measured_device("t|fast", Device::Metal, fake(1.0, 2.0)),
            Device::Metal,
            "2× faster: GPU"
        );
        assert_eq!(
            measured_device("t|close", Device::Metal, fake(1.0, 1.4)),
            Device::Cpu,
            "1.4×: stays on CPU"
        );
        assert_eq!(
            measured_device("t|slow", Device::Metal, fake(3.0, 1.0)),
            Device::Cpu,
            "slower: CPU"
        );
        assert_eq!(
            measured_device("t|nogpu", Device::Metal, |d: Device| matches!(
                d,
                Device::Cpu
            )
            .then_some(1.0)),
            Device::Cpu
        );
        // remembered in-process: the timer is not called again
        let mut calls = 0;
        assert_eq!(
            measured_device("t|fast", Device::Metal, |_| {
                calls += 1;
                Some(9.0)
            }),
            Device::Metal
        );
        assert_eq!(calls, 0);
        // and on disk, for the next run
        assert_eq!(placement_cache_get("t|fast"), Some(Device::Metal));
        assert_eq!(placement_cache_get("t|slow"), Some(Device::Cpu));
        // the CPU is never probed
        assert_eq!(
            measured_device("t|cpu", Device::Cpu, |_| panic!("no probe for the CPU")),
            Device::Cpu
        );
        // a decision is final for the process, whatever the cost model says
        // later: a CPU pick stays CPU when the next pick is a GPU (no probe),
        // and a measured GPU stays GPU when the next pick is the CPU
        assert_eq!(
            measured_device("t|cpu", Device::Metal, |_| panic!("decided already")),
            Device::Cpu
        );
        assert_eq!(
            measured_device("t|fast", Device::Cpu, |_| panic!("decided already")),
            Device::Metal
        );
        // SAFETY: as above.
        unsafe { std::env::remove_var("RLX_PLACEMENT_CACHE") };
        rlx_ir::env::reload();
        let _ = std::fs::remove_file(&path);
    }
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// At most `gpu_jobs()` threads hold a GPU slot at once; a nested
    /// acquisition on the same thread does not wait on itself; CPU work never
    /// takes a slot.
    #[test]
    fn gpu_slots_cap_concurrency_and_nest() {
        let (live, peak) = (AtomicUsize::new(0), AtomicUsize::new(0));
        std::thread::scope(|s| {
            for _ in 0..8 {
                s.spawn(|| {
                    let _a = gpu_slot(Device::Metal);
                    let _nested = gpu_slot(Device::Metal); // must not deadlock
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    live.fetch_sub(1, Ordering::SeqCst);
                });
            }
            // CPU work is not counted: 8 CPU holders at once
            let cpu: Vec<_> = (0..8).map(|_| gpu_slot(Device::Cpu)).collect();
            drop(cpu);
        });
        let p = peak.load(Ordering::SeqCst);
        if let Some(cap) = gpu_jobs() {
            assert!(p >= 1 && p <= cap, "peak {p} > cap {cap}");
        }
    }
}
