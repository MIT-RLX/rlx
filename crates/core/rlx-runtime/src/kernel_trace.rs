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

//! Compile-time gated kernel tracing (plan #7).
//!
//! Borrowed from MAX's `Trace, TraceLevel, trace_arg` pattern.
//! Tracing calls are baked into hot paths but the *macro* expands to
//! either a stamped `eprintln!` (when the `kernel-trace` feature is
//! on) or to nothing (default). The compiler eliminates the
//! disabled branch entirely — production builds pay zero overhead.
//!
//! Use it like:
//! ```ignore
//! rlx_runtime::ktrace!("matmul", "m={m} k={k} n={n}");
//! ```
//!
//! The macro takes a `kind` (op/section name) and a format string
//! plus args. Output is namespaced with `[ktrace:<kind>]` and
//! includes a monotonic timestamp from `rlx_ir::Tick`.

#[cfg(feature = "kernel-trace")]
#[doc(hidden)]
pub fn _emit(kind: &str, msg: std::fmt::Arguments<'_>) {
    use std::sync::OnceLock;
    static T0: OnceLock<rlx_ir::Tick> = OnceLock::new();
    let start = *T0.get_or_init(rlx_ir::Tick::now);
    let now = rlx_ir::Tick::now();
    eprintln!("[ktrace:{kind}] +{:>10} ns  {msg}", now.elapsed_ns(start));
}

#[cfg(not(feature = "kernel-trace"))]
#[doc(hidden)]
pub fn _emit(_kind: &str, _msg: std::fmt::Arguments<'_>) {}

/// Compile-time gated kernel trace. Expands to a no-op call without
/// the `kernel-trace` feature; the optimizer removes it entirely.
#[macro_export]
macro_rules! ktrace {
    ($kind:expr, $($arg:tt)+) => {{
        $crate::kernel_trace::_emit($kind, format_args!($($arg)+));
    }};
}

#[cfg(test)]
mod tests {
    #[test]
    fn macro_compiles_and_runs() {
        // The macro must compile and run regardless of feature state.
        // With the feature off, this is a no-op call — verify it
        // doesn't panic and doesn't write anything we can observe in
        // a unit test.
        crate::ktrace!("test", "x={} y={}", 1, 2);
    }
}
