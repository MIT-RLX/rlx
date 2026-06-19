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

//! Cycle-accurate timing primitive (#66 in plan.md).
//!
//! `Instant::now()` on macOS already wraps `mach_continuous_time` (which on
//! Apple Silicon ultimately reads `CNTVCT_EL0`), so the wall-clock precision
//! is fine. The win from going direct is two-fold:
//!
//!   1. **Resolution.** `Instant` exposes nanoseconds via `Duration`, but the
//!      hardware tick is ~41 ns on M-series (24 MHz `CNTFRQ_EL0`). Tracking
//!      raw ticks lets the autotuner reason at the actual hardware grain
//!      instead of pretending it has 1 ns precision.
//!   2. **Overhead.** `Instant::now()` is a few hundred cycles of wrapper
//!      and `Duration` math. The raw `mrs` is one instruction. For
//!      sub-microsecond kernels (small-tile probes in `calibrate.rs`) the
//!      wrapper itself becomes a measurable fraction of the timed region.
//!
//! Falls back to `Instant` on non-AArch64 / non-Apple targets so the API
//! stays portable.

#[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
use std::time::Instant;

/// Opaque tick reading. Subtract two of these to get a `Duration`.
#[derive(Copy, Clone, Debug)]
pub struct Tick {
    #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
    cycles: u64,
    #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
    instant: Instant,
}

impl Tick {
    /// Read the current tick.
    #[inline(always)]
    pub fn now() -> Self {
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        {
            Tick {
                cycles: read_cntvct(),
            }
        }
        #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
        {
            Tick {
                instant: Instant::now(),
            }
        }
    }

    /// Elapsed nanoseconds since `start`. Saturates at zero if the clock
    /// went backwards (it shouldn't, but the kernel is allowed to lie).
    #[inline(always)]
    pub fn elapsed_ns(&self, start: Tick) -> u64 {
        #[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
        {
            let dt = self.cycles.saturating_sub(start.cycles);
            // CNTFRQ_EL0 is 24 MHz on Apple Silicon → 1 tick = 1000/24 ns.
            // Cached the first time we ask; the value never changes.
            let freq = cntfrq_hz();
            ((dt as u128) * 1_000_000_000u128 / freq as u128) as u64
        }
        #[cfg(not(all(target_arch = "aarch64", target_vendor = "apple")))]
        {
            self.instant.duration_since(start.instant).as_nanos() as u64
        }
    }

    #[inline(always)]
    pub fn elapsed_us(&self, start: Tick) -> f64 {
        self.elapsed_ns(start) as f64 / 1_000.0
    }

    #[inline(always)]
    pub fn elapsed_ms(&self, start: Tick) -> f64 {
        self.elapsed_ns(start) as f64 / 1_000_000.0
    }
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
#[inline(always)]
fn read_cntvct() -> u64 {
    let val: u64;
    // Safety: `mrs cntvct_el0, X` is unprivileged on AArch64 and reads the
    // virtual count register. No memory effects.
    unsafe {
        std::arch::asm!("mrs {0}, cntvct_el0", out(reg) val, options(nomem, nostack));
    }
    val
}

#[cfg(all(target_arch = "aarch64", target_vendor = "apple"))]
fn cntfrq_hz() -> u64 {
    use std::sync::OnceLock;
    static FREQ: OnceLock<u64> = OnceLock::new();
    *FREQ.get_or_init(|| {
        let val: u64;
        unsafe {
            std::arch::asm!("mrs {0}, cntfrq_el0", out(reg) val, options(nomem, nostack));
        }
        // Apple Silicon reports 24 MHz; guard against bogus zero just in case.
        if val == 0 { 24_000_000 } else { val }
    })
}

/// Time `f`, returning `(result, elapsed_ns)`. Inlined so the surrounding
/// loop can keep the closure body in registers.
#[inline(always)]
pub fn time_ns<R>(f: impl FnOnce() -> R) -> (R, u64) {
    let t0 = Tick::now();
    let r = f();
    let t1 = Tick::now();
    (r, t1.elapsed_ns(t0))
}

/// Cache-busting buffer — sized to evict L1+L2 on Apple Silicon
/// (M-series: 192 KB L1d / core, 16 MB L2 shared per cluster).
/// Borrowed from MAX's `internal_utils/_cache_busting.mojo` (#19).
///
/// Allocate once, then call `.thrash()` between bench iterations to
/// flush whatever the previous iteration left in cache. Without this,
/// "cache-cold" timings actually measure cache-warm performance and
/// over-report by 2-5×.
pub struct CacheBuster {
    buf: Vec<u8>,
}

impl CacheBuster {
    /// Allocate a buster sized to evict the targeted cache. Defaults
    /// to 32 MB — twice the M-series L2 — which guarantees full L2
    /// eviction. Pass a custom size for finer control (e.g. 256 KB
    /// to evict only L1).
    pub fn new() -> Self {
        Self::with_bytes(32 * 1024 * 1024)
    }

    pub fn with_bytes(bytes: usize) -> Self {
        Self {
            buf: vec![0u8; bytes],
        }
    }

    /// Walk the buffer once, touching every cache line. After this
    /// returns, the previous workload's data is evicted.
    #[inline(never)]
    pub fn thrash(&mut self) {
        // 64-byte stride matches the cacheline size on Apple Silicon.
        // Use a volatile-ish read+write so the optimizer can't elide.
        let len = self.buf.len();
        let ptr = self.buf.as_mut_ptr();
        let mut acc: u8 = 0;
        let mut i = 0usize;
        while i < len {
            unsafe {
                let p = ptr.add(i);
                acc = acc.wrapping_add(std::ptr::read_volatile(p));
                std::ptr::write_volatile(p, acc);
            }
            i += 64;
        }
        // Write the accumulator somewhere observable so dead-store
        // elimination doesn't drop the loop on aggressive opt levels.
        std::hint::black_box(acc);
    }
}

impl Default for CacheBuster {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_is_monotonic() {
        // CNTVCT_EL0 ticks at ~24 MHz on Apple Silicon → ~41 ns per tick.
        // Two back-to-back reads can land on the same tick. Sleep one
        // tick period so the delta is guaranteed non-zero.
        let a = Tick::now();
        std::thread::sleep(std::time::Duration::from_micros(50));
        let b = Tick::now();
        assert!(b.elapsed_ns(a) > 0);
    }

    #[test]
    fn elapsed_units_agree() {
        let a = Tick::now();
        std::thread::sleep(std::time::Duration::from_millis(2));
        let b = Tick::now();
        let ns = b.elapsed_ns(a);
        assert!(ns >= 1_500_000, "expected >=1.5ms, got {ns}ns");
        assert!((b.elapsed_ms(a) - ns as f64 / 1e6).abs() < 1e-6);
    }
}
