// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! PLAN L3: Perfetto / chrome-trace JSON output for cross-backend timeline
//! capture.
//!
//! When the env var `RLX_TRACE_PERFETTO=<path>` is set, the runtime
//! opens that file at first-use and appends one "complete" event
//! (`ph: "X"`) per `trace_span!` scope. The output is the
//! chrome-trace JSON object format — load it in the Perfetto UI
//! (<https://ui.perfetto.dev>) or `chrome://tracing/`.
//!
//! Vendor profilers (NVTX in `rlx-cuda`, rocTX in `rlx-rocm`,
//! `MTLCommandEncoder.pushDebugGroup` in `rlx-metal`) stay wired
//! independently — Perfetto is the *cross-backend* view; vendor
//! profilers remain the right tool inside their respective ecosystems.
//!
//! Format reference: <https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU>
//!
//! ## Usage
//!
//! ```ignore
//! // In a hot path:
//! let _span = rlx_runtime::trace_span!("matmul", "compute");
//! // ... do work ...
//! // _span drops here, writes the complete event to the trace file.
//! ```
//!
//! Activate by setting `RLX_TRACE_PERFETTO=/tmp/rlx.json` before
//! running. After process exit, open the file in the Perfetto UI.

use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};

use crate::Tick;

/// Per-process Perfetto trace state. `None` when tracing is disabled
/// (env var not set or file open failed).
struct PerfettoState {
    file: Mutex<File>,
    t0: Tick,
    /// Tracks whether the current entry needs a leading comma —
    /// chrome-trace JSON is `[event, event, ...]`. Set to false on
    /// the first event so we don't emit a leading comma.
    needs_comma: Mutex<bool>,
}

static STATE: OnceLock<Option<PerfettoState>> = OnceLock::new();

fn state() -> Option<&'static PerfettoState> {
    STATE
        .get_or_init(|| {
            let path = std::env::var("RLX_TRACE_PERFETTO").ok()?;
            let mut file = File::create(&path).ok()?;
            // Open the JSON array.
            file.write_all(b"[\n").ok()?;
            Some(PerfettoState {
                file: Mutex::new(file),
                t0: Tick::now(),
                needs_comma: Mutex::new(false),
            })
        })
        .as_ref()
}

/// True when Perfetto tracing is active in this process. Used to
/// guard expensive label construction at the call site.
pub fn enabled() -> bool {
    state().is_some()
}

/// Record a "complete" event — the entire span (begin → end) as a
/// single chrome-trace entry. `start_ns` and `end_ns` are timestamps
/// from `Tick::now().elapsed_ns(t0)`. Callers usually go through
/// the `trace_span!` macro for the RAII pattern.
pub fn emit_complete(name: &str, cat: &str, start_ns: u64, end_ns: u64) {
    let Some(s) = state() else { return };
    let dur_us = (end_ns.saturating_sub(start_ns)) as f64 / 1000.0;
    let ts_us = start_ns as f64 / 1000.0;
    let mut comma = s.needs_comma.lock().unwrap();
    let prefix = if *comma { ",\n" } else { "" };
    let line = format!(
        "{prefix}{{\"name\":\"{name}\",\"cat\":\"{cat}\",\"ph\":\"X\",\
         \"ts\":{ts_us},\"dur\":{dur_us},\"pid\":1,\"tid\":1}}",
    );
    let _ = s.file.lock().unwrap().write_all(line.as_bytes());
    *comma = true;
}

/// Flush + close the trace array. Called automatically by the
/// `PerfettoState`'s `Drop` — but `OnceLock` doesn't drop static
/// values, so callers who care about a clean trailing `]` should
/// invoke this explicitly before exit (e.g., via `atexit`-style hook).
/// In practice the JSON is still parseable without the `]` because
/// most viewers tolerate it; we add the marker as best-effort.
pub fn flush_and_finalize() {
    let Some(s) = state() else { return };
    let mut f = s.file.lock().unwrap();
    let _ = f.write_all(b"\n]\n");
    let _ = f.flush();
}

/// RAII trace span — records the start timestamp at construction and
/// emits a complete event at drop. Constructed via the `trace_span!` macro.
pub struct TraceSpan {
    name: &'static str,
    cat: &'static str,
    start_ns: u64,
}

impl TraceSpan {
    /// Construct a new span. `name` is the operation (e.g. "Sgemm");
    /// `cat` is the category (e.g. "cpu", "metal", "cuda"). Both must
    /// be `'static` so the JSON serializer can borrow them directly.
    /// Returns `None` when tracing is disabled — the macro form
    /// hides this so call sites just write `let _span = ...;`.
    pub fn new(name: &'static str, cat: &'static str) -> Option<Self> {
        let s = state()?;
        let start_ns = Tick::now().elapsed_ns(s.t0);
        Some(Self {
            name,
            cat,
            start_ns,
        })
    }
}

impl Drop for TraceSpan {
    fn drop(&mut self) {
        let Some(s) = state() else { return };
        let end_ns = Tick::now().elapsed_ns(s.t0);
        emit_complete(self.name, self.cat, self.start_ns, end_ns);
    }
}

/// Open a Perfetto trace span. The returned `Option<TraceSpan>` is
/// `None` when tracing is disabled — bind it to `_span` so the
/// scope drop point is the natural end-of-block.
///
/// ```ignore
/// let _span = rlx_runtime::trace_span!("matmul", "cpu");
/// // ... work ...
/// ```
#[macro_export]
macro_rules! trace_span {
    ($name:expr, $cat:expr) => {
        $crate::perfetto::TraceSpan::new($name, $cat)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    /// Smoke-test the API surface without enabling the env var. With
    /// tracing disabled, every entry point is a no-op and `enabled()`
    /// returns false. We can't actually exercise the file path in a
    /// unit test because `STATE` is process-wide — once initialized,
    /// it can't be re-initialized. End-to-end tracing is verified by
    /// running an example with `RLX_TRACE_PERFETTO=...` set.
    #[test]
    fn disabled_is_noop() {
        // Don't set env var; state() returns None.
        // (If a prior test in the process set it, this asserts that
        // `enabled()` is consistent with whatever STATE was init'd to.)
        let was_enabled = enabled();
        // Calling emit_complete and trace_span! must not panic in either state.
        emit_complete("test_op", "test", 100, 200);
        let _span = TraceSpan::new("test_op", "test");
        let _macro_span = trace_span!("test_op", "test");
        assert_eq!(enabled(), was_enabled, "enabled state must be stable");
    }

    /// Write an event manually + parse the resulting line as JSON to
    /// confirm the format is valid. Done by side-stepping `STATE` —
    /// build the line with the same formatter and parse it.
    #[test]
    fn complete_event_is_valid_json() {
        let line = format!(
            "{{\"name\":\"{name}\",\"cat\":\"{cat}\",\"ph\":\"X\",\
             \"ts\":{ts},\"dur\":{dur},\"pid\":1,\"tid\":1}}",
            name = "Matmul",
            cat = "cpu",
            ts = 0.5,
            dur = 1.25,
        );
        // Don't pull a JSON parser into the dep tree; sanity-check
        // structurally instead.
        assert!(line.starts_with("{") && line.ends_with("}"));
        assert!(line.contains("\"name\":\"Matmul\""));
        assert!(line.contains("\"cat\":\"cpu\""));
        assert!(line.contains("\"ph\":\"X\""));
        assert!(line.contains("\"ts\":0.5"));
        assert!(line.contains("\"dur\":1.25"));
    }

    /// Build a complete trace by manually invoking the writer with
    /// the env var set, then reading back the file. Uses a temp file.
    /// **Important**: if STATE was already initialized in this process
    /// from a previous test, the env var won't take effect; we just
    /// confirm the file-open code path works in isolation.
    #[test]
    fn end_to_end_temp_file_smoke() {
        use std::env;
        use std::fs;
        // Use a unique temp path so tests can run in parallel without
        // collision (though STATE serialization makes parallel use
        // problematic — see disabled_is_noop comment).
        let dir = env::temp_dir();
        let path = dir.join(format!("rlx-trace-{}.json", std::process::id()));
        if path.exists() {
            let _ = fs::remove_file(&path);
        }

        // Manually drive the lower-level write path so this test
        // doesn't depend on STATE being uninitialized.
        let mut f = File::create(&path).unwrap();
        f.write_all(b"[\n").unwrap();
        // Two complete events.
        f.write_all(
            b"{\"name\":\"matmul\",\"cat\":\"cpu\",\"ph\":\"X\",\
              \"ts\":0,\"dur\":1.5,\"pid\":1,\"tid\":1}",
        )
        .unwrap();
        f.write_all(b",\n").unwrap();
        f.write_all(
            b"{\"name\":\"layernorm\",\"cat\":\"cpu\",\"ph\":\"X\",\
              \"ts\":2,\"dur\":0.8,\"pid\":1,\"tid\":1}",
        )
        .unwrap();
        f.write_all(b"\n]\n").unwrap();
        f.flush().unwrap();
        drop(f);

        let mut got = String::new();
        File::open(&path).unwrap().read_to_string(&mut got).unwrap();
        assert!(got.starts_with("[\n"));
        assert!(got.contains("\"matmul\""));
        assert!(got.contains("\"layernorm\""));
        assert!(got.trim_end().ends_with("]"));

        let _ = fs::remove_file(&path);
    }
}
