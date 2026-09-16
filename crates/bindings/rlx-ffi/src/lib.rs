// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! C ABI for the RLX **distributed node**.
//!
//! Lets a non-Rust shell join an RLX mesh as a worker rank: an iOS app (link
//! the `staticlib` into an `.xcframework`), an embedded Linux host driving an
//! FPGA board, or anything else that speaks C.
//!
//! The node runs on its own thread — every function here returns immediately,
//! because a serving loop parks in `recv` between activations and a UI thread
//! must not block on it.
//!
//! ```c
//! rlx_node_start(1, 2, "192.168.1.10:29500,192.168.1.11:29501", "auto");
//! char buf[256];
//! rlx_node_status(buf, sizeof buf);   // "running"
//! rlx_node_stop();                     // cooperative
//! ```
//!
//! # Threading
//!
//! All entry points are safe to call from any thread; state is behind a mutex.
//! Only one node may run per process.

use rlx_runtime::dist::node::{
    NodeConfig, NodeControl, NodeStopHandle, serve_trainer_here, serve_worker,
};
use std::ffi::{CStr, CString, c_char, c_int};
use std::sync::{Mutex, OnceLock, mpsc};

/// Result codes. Negative values are errors.
pub const RLX_NODE_OK: c_int = 0;
/// A node is already running in this process.
pub const RLX_NODE_ERR_BUSY: c_int = -1;
/// A pointer argument was null, or a string was not valid UTF-8.
pub const RLX_NODE_ERR_ARG: c_int = -2;
/// `rank`/`world` were inconsistent, or the peer list did not match `world`.
pub const RLX_NODE_ERR_CONFIG: c_int = -3;
/// The serving thread could not be spawned.
pub const RLX_NODE_ERR_SPAWN: c_int = -4;
/// `mode` was not one of the recognised values.
pub const RLX_NODE_ERR_MODE: c_int = -5;

/// Weight/data URIs are resolved **on the node**, so nothing but the spec
/// crosses the wire. The built-in schemes (`file://`, `gguf://`,
/// `safetensors://`) are handled inside `run_train`/`recv_stage`; this is the
/// fallback for anything else, and an empty vector is the honest answer — the
/// caller sees it as a parameter that failed to load rather than as zeros that
/// look like data.
fn uri_resolver(_uri: &str) -> Vec<f32> {
    Vec::new()
}

struct Slot {
    stop: NodeStopHandle,
    done: mpsc::Receiver<String>,
    last: Option<String>,
}

fn slot() -> &'static Mutex<Option<Slot>> {
    static S: OnceLock<Mutex<Option<Slot>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(None))
}

/// Last error message, so a caller that got a code can read the detail.
fn last_error() -> &'static Mutex<String> {
    static E: OnceLock<Mutex<String>> = OnceLock::new();
    E.get_or_init(|| Mutex::new(String::new()))
}

fn set_error(msg: impl Into<String>) {
    if let Ok(mut e) = last_error().lock() {
        *e = msg.into();
    }
}

/// # Safety
/// `p` must be null or a valid NUL-terminated C string.
unsafe fn cstr(p: *const c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .ok()
        .map(str::to_owned)
}

/// Copy `s` into `buf` (NUL-terminated, truncated to fit). Returns the length
/// written, excluding the terminator, or `RLX_NODE_ERR_ARG`.
///
/// # Safety
/// `buf` must be writable for `cap` bytes.
unsafe fn fill(buf: *mut c_char, cap: usize, s: &str) -> c_int {
    if buf.is_null() || cap == 0 {
        return RLX_NODE_ERR_ARG;
    }
    let Ok(c) = CString::new(s) else {
        return RLX_NODE_ERR_ARG;
    };
    let bytes = c.as_bytes_with_nul();
    let n = bytes.len().min(cap);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr().cast::<c_char>(), buf, n);
        // Truncation still leaves a valid C string.
        *buf.add(n - 1) = 0;
    }
    (n - 1) as c_int
}

/// Join a mesh as worker `rank` of `world`.
///
/// * `peers` — comma-separated `host:port` list indexed by rank. Pass an empty
///   string to find the coordinator by UDP broadcast instead.
/// * `device` — `"auto"` or a backend name (`"cpu"`, `"metal"`, `"gpu"`, …).
/// * `mode` — `"infer"` to serve a shipped inference stage, or `"train"` to
///   join a data-parallel training run. `""` means `"infer"`.
///
/// Returns [`RLX_NODE_OK`] once the serving thread is spawned — *not* once the
/// mesh is joined. Poll [`rlx_node_status`] for that.
///
/// # Training and backgrounding
///
/// A training rank cannot drop out partway: the gradient reduce is a barrier,
/// so a node that stops stalls every other rank. [`rlx_node_stop`] is honoured
/// between inference activations but **not** mid-training-run. Start a training
/// node only when the process will stay alive for it.
///
/// # Safety
/// `peers`, `device` and `mode` must be valid NUL-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_node_start(
    rank: c_int,
    world: c_int,
    peers: *const c_char,
    device: *const c_char,
    mode: *const c_char,
) -> c_int {
    let (Some(peers), Some(device), Some(mode)) =
        (unsafe { cstr(peers) }, unsafe { cstr(device) }, unsafe {
            cstr(mode)
        })
    else {
        set_error("peers/device/mode must be valid UTF-8 C strings");
        return RLX_NODE_ERR_ARG;
    };
    let training = match mode.as_str() {
        "" | "infer" => false,
        "train" => true,
        other => {
            set_error(format!("unknown mode [{other}]; expected infer or train"));
            return RLX_NODE_ERR_MODE;
        }
    };
    let Ok(mut g) = slot().lock() else {
        set_error("node state poisoned");
        return RLX_NODE_ERR_BUSY;
    };
    if g.as_ref().is_some_and(|s| s.last.is_none()) {
        set_error("a node is already running in this process");
        return RLX_NODE_ERR_BUSY;
    }
    if rank < 0 || world < 1 || rank >= world {
        set_error(format!("bad rank/world: {rank}/{world}"));
        return RLX_NODE_ERR_CONFIG;
    }

    let addrs: Vec<String> = peers
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Topology follows from what the caller supplied. A phone is a star
    // worker: it dials the coordinator and nothing dials it back (most Wi-Fi
    // will not route inbound to a handset), so a lone address — or discovery,
    // where the coordinator announces itself — means star. A full per-rank
    // list means the caller wants a mesh and believes every rank is reachable.
    let base = NodeConfig::new(rank as u32, world as u32).device(device);
    let cfg = if addrs.is_empty() {
        base.star().discover(29600, 29500)
    } else {
        let star = addrs.len() == 1 && world > 1;
        let base = if star { base.star() } else { base.mesh() };
        match base.peers(addrs) {
            Ok(c) => c,
            Err(e) => {
                set_error(e);
                return RLX_NODE_ERR_CONFIG;
            }
        }
    };

    let ctl = NodeControl::unbounded();
    let stop = ctl.stop_handle();
    let (tx, rx) = mpsc::channel();

    if let Err(e) = std::thread::Builder::new()
        .name("rlx-node".into())
        .spawn(move || {
            let msg = match cfg.connect() {
                Err(e) => format!("error: connect failed: {e}"),
                Ok(group) if training => match serve_trainer_here(&group, uri_resolver, false) {
                    Ok(r) => format!(
                        "ok: rank {} trained on {} ({}), {} sample(s), loss {:.4}->{:.4}",
                        r.rank,
                        r.metrics.device.name(),
                        r.platform,
                        r.metrics.samples,
                        r.metrics.first_loss,
                        r.metrics.last_loss
                    ),
                    Err(e) => format!("error: {e}"),
                },
                Ok(group) => match serve_worker(&group, uri_resolver) {
                    Ok(r) => format!(
                        "ok: rank {} on {} ({}), {} activation(s)",
                        r.rank,
                        r.device.name(),
                        r.platform,
                        r.activations
                    ),
                    Err(e) => format!("error: {e}"),
                },
            };
            let _ = tx.send(msg);
        })
    {
        set_error(format!("spawn: {e}"));
        return RLX_NODE_ERR_SPAWN;
    }

    *g = Some(Slot {
        stop,
        done: rx,
        last: None,
    });
    RLX_NODE_OK
}

/// Write the node state into `buf`: `idle` | `running` | `stopping` |
/// `ok: …` | `error: …`. Returns the length written, or a negative error code.
///
/// # Safety
/// `buf` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_node_status(buf: *mut c_char, cap: usize) -> c_int {
    let Ok(mut g) = slot().lock() else {
        return unsafe { fill(buf, cap, "error: node state poisoned") };
    };
    let s = match g.as_mut() {
        None => "idle".to_string(),
        Some(s) => {
            if s.last.is_none()
                && let Ok(msg) = s.done.try_recv()
            {
                s.last = Some(msg);
            }
            match &s.last {
                Some(m) => m.clone(),
                None if s.stop.is_stopped() => "stopping".into(),
                None => "running".into(),
            }
        }
    };
    unsafe { fill(buf, cap, &s) }
}

/// Ask the node to leave the mesh after its current activation.
///
/// Cooperative: a node parked in `recv` exits when its peer sends or the link
/// drops, so [`rlx_node_status`] may report `running` briefly afterwards.
#[unsafe(no_mangle)]
pub extern "C" fn rlx_node_stop() -> c_int {
    let Ok(g) = slot().lock() else {
        return RLX_NODE_ERR_BUSY;
    };
    match g.as_ref() {
        Some(s) => {
            s.stop.stop();
            RLX_NODE_OK
        }
        None => RLX_NODE_OK,
    }
}

/// Write the last error detail into `buf`. Returns the length written.
///
/// # Safety
/// `buf` must be writable for `cap` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rlx_node_last_error(buf: *mut c_char, cap: usize) -> c_int {
    let msg = last_error().lock().map(|e| e.clone()).unwrap_or_default();
    unsafe { fill(buf, cap, &msg) }
}

/// Platform tag this library was built for (`ios`, `android`, `linux`, …).
/// Returns a static NUL-terminated string; the caller must not free it.
#[unsafe(no_mangle)]
pub extern "C" fn rlx_node_platform() -> *const c_char {
    // A `const` tag, so one CString per process is enough.
    static P: OnceLock<CString> = OnceLock::new();
    P.get_or_init(|| CString::new(rlx_runtime::dist::node::platform_tag()).unwrap_or_default())
        .as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    /// Serializes every test that touches the node C ABI.
    ///
    /// The ABI is built on process-global state — one node slot and one
    /// last-error string — which is correct for a C consumer but means a test
    /// that calls `rlx_node_start` and *then* reads `rlx_node_last_error` can
    /// read a sibling's message instead of its own. That is not hypothetical:
    /// `rejects_bad_rank_world` failed with `unknown mode [trian]`, the string
    /// `rejects_an_unknown_mode` had just written. Hold this for any test here
    /// that calls an `rlx_node_*` entry point other than `rlx_node_platform`.
    static FFI_TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Take [`FFI_TEST_LOCK`], ignoring poisoning — it orders calls, and guards
    /// no invariant a panicking test could leave broken.
    fn ffi_lock() -> std::sync::MutexGuard<'static, ()> {
        FFI_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn status() -> String {
        let mut buf = [0 as c_char; 256];
        let n = unsafe { rlx_node_status(buf.as_mut_ptr(), buf.len()) };
        assert!(n >= 0, "status returned {n}");
        unsafe { CStr::from_ptr(buf.as_ptr()) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn rejects_bad_rank_world() {
        let _lock = ffi_lock();
        let peers = CString::new("127.0.0.1:1").unwrap();
        let dev = CString::new("cpu").unwrap();
        // rank >= world
        let mode = CString::new("infer").unwrap();
        let rc = unsafe { rlx_node_start(2, 2, peers.as_ptr(), dev.as_ptr(), mode.as_ptr()) };
        assert_eq!(rc, RLX_NODE_ERR_CONFIG);

        let mut buf = [0 as c_char; 256];
        let n = unsafe { rlx_node_last_error(buf.as_mut_ptr(), buf.len()) };
        assert!(n > 0);
        let msg = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy();
        assert!(msg.contains("2/2"), "unexpected error: {msg}");
    }

    #[test]
    fn rejects_peer_count_mismatch() {
        let _lock = ffi_lock();
        // 2 addresses for world 3 — neither a full mesh nor a star's lone
        // coordinator, so it is caught before any socket work.
        let peers = CString::new("127.0.0.1:1,127.0.0.1:2").unwrap();
        let dev = CString::new("cpu").unwrap();
        let mode = CString::new("infer").unwrap();
        let rc = unsafe { rlx_node_start(1, 3, peers.as_ptr(), dev.as_ptr(), mode.as_ptr()) };
        assert_eq!(rc, RLX_NODE_ERR_CONFIG);
    }

    #[test]
    fn rejects_an_unknown_mode() {
        let _lock = ffi_lock();
        let peers = CString::new("127.0.0.1:1").unwrap();
        let dev = CString::new("cpu").unwrap();
        let mode = CString::new("trian").unwrap(); // a plausible typo
        let rc = unsafe { rlx_node_start(0, 1, peers.as_ptr(), dev.as_ptr(), mode.as_ptr()) };
        assert_eq!(rc, RLX_NODE_ERR_MODE);
        let mut buf = [0 as c_char; 256];
        let n = unsafe { rlx_node_last_error(buf.as_mut_ptr(), buf.len()) };
        assert!(n > 0);
        let msg = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy();
        assert!(
            msg.contains("trian"),
            "error should name the bad mode: {msg}"
        );
    }

    #[test]
    fn rejects_null_strings() {
        let _lock = ffi_lock();
        let dev = CString::new("cpu").unwrap();
        let mode = CString::new("infer").unwrap();
        let rc = unsafe { rlx_node_start(0, 1, std::ptr::null(), dev.as_ptr(), mode.as_ptr()) };
        assert_eq!(rc, RLX_NODE_ERR_ARG);
    }

    #[test]
    fn status_is_idle_before_start() {
        let _lock = ffi_lock();
        // These tests share one process-global slot. `idle` holds because
        // every start in this module fails validation, and the lock keeps this
        // from observing one mid-flight.
        assert_eq!(status(), "idle");
    }

    #[test]
    fn stop_is_safe_with_no_node() {
        let _lock = ffi_lock();
        assert_eq!(rlx_node_stop(), RLX_NODE_OK);
    }

    #[test]
    fn platform_tag_is_nonempty() {
        let p = rlx_node_platform();
        assert!(!p.is_null());
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy();
        assert!(!s.is_empty() && s != "unknown", "platform tag: {s}");
    }

    #[test]
    fn status_buffer_truncates_safely() {
        let _lock = ffi_lock();
        let mut buf = [0 as c_char; 3];
        let n = unsafe { rlx_node_status(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(n, 2);
        let s = unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy();
        assert_eq!(s, "id"); // "idle" truncated, still NUL-terminated
    }
}
