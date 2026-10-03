// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-time target predicates (plan #78).
//!
//! Borrowed from MAX's `std.gpu.host.info` `is_cpu` / `is_valid_target`
//! pattern. Centralized const fns the optimizer / fusion patterns use
//! to prune backend-specific code paths at compile time instead of
//! branching at runtime.
//!
//! Why a module instead of inline `cfg!()`?
//!   - One source of truth: changing a predicate (e.g. broadening
//!     `has_amx` to include M-series Pro/Max) updates every call site.
//!   - Documented contract: each predicate's docstring records what
//!     the answer means and what kernels rely on it.
//!   - `const fn`: can be used in const contexts (lookup tables,
//!     static assertions).

/// True on Apple Silicon — M-series Macs and the A/M/S-series chips in
/// iPhones, iPads, Apple TVs, Apple Watches and Vision Pro (plus the
/// aarch64 simulators). They ship the AMX coprocessor (and, except the
/// Watch, a Metal GPU), so the same fast paths apply. The x86_64
/// simulators are deliberately excluded (they run on Intel Macs).
pub const fn is_apple_silicon() -> bool {
    cfg!(all(target_arch = "aarch64", target_vendor = "apple"))
}

/// True on iOS (iPhone/iPad), device or simulator.
pub const fn is_ios() -> bool {
    cfg!(target_os = "ios")
}

/// True on tvOS (Apple TV), device or simulator.
pub const fn is_tvos() -> bool {
    cfg!(target_os = "tvos")
}

/// True on watchOS (Apple Watch), device or simulator. Note: watchOS has
/// no Metal — see [`has_metal`].
pub const fn is_watchos() -> bool {
    cfg!(target_os = "watchos")
}

/// True on visionOS (Apple Vision Pro / "xrOS"), device or simulator.
pub const fn is_visionos() -> bool {
    cfg!(target_os = "visionos")
}

/// True on any Apple platform that RLX builds native backends for —
/// macOS, iOS, tvOS, watchOS or visionOS. The Accelerate (AMX BLAS) and
/// CoreML/ANE backends compile on all of them; use this instead of
/// repeating the `target_vendor = "apple"` cfg inline. (Metal is the one
/// exception — it skips watchOS; see [`has_metal`].)
pub const fn is_apple_platform() -> bool {
    cfg!(target_vendor = "apple")
}

/// True on aarch64 broadly (Apple Silicon + Linux ARM + AWS Graviton).
pub const fn is_aarch64() -> bool {
    cfg!(target_arch = "aarch64")
}

/// True on x86_64 (Intel / AMD desktop and server).
pub const fn is_x86_64() -> bool {
    cfg!(target_arch = "x86_64")
}

/// True on macOS (any arch).
pub const fn is_macos() -> bool {
    cfg!(target_os = "macos")
}

/// True if NEON intrinsics are available. On AArch64 NEON is
/// architectural (always present); on x86 we'd need explicit
/// runtime detection (not done here).
pub const fn has_neon() -> bool {
    cfg!(target_arch = "aarch64")
}

/// True only when this build was *pinned* to AVX2 at compile time, i.e. it
/// was given `-C target-cpu=native` (or `x86-64-v3`, or an explicit
/// `+avx2`). It is NOT "this CPU has AVX2": a stock `cargo build` answers
/// false on an AVX-512 server, because a stock build deliberately stays
/// baseline so one artifact runs everywhere.
///
/// Do not use this to pick a code path. It reads like a capability check and
/// is really a build-flag check, and acting on it is how a kernel ends up
/// existing only in a non-portable artifact: `rlx-cpu`'s `bias_gelu` and
/// `layer_norm_row` were gated this way, so the portable build silently lost
/// their AVX2 kernels and the only way to get them was a binary that
/// `SIGILL`s on any pre-Gracemont Atom. Both now dispatch on
/// `is_x86_feature_detected!`, which is the pattern to copy — compile the
/// path unconditionally, choose it at runtime. `just check-isa` enforces it.
#[deprecated(
    since = "0.2.17",
    note = "a build-flag check masquerading as a capability check; dispatch on \
            `std::arch::is_x86_feature_detected!(\"avx2\")` at runtime instead"
)]
pub const fn has_avx2_likely() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx2"))
}

/// True when the target has access to the Apple AMX coprocessor —
/// i.e. through Accelerate's BLAS path. Not a direct AMX intrinsic
/// gate (those are unstable / undocumented); used by callers that
/// want to know "is there something matmul-grade besides NEON
/// available."
pub const fn has_amx() -> bool {
    is_apple_silicon()
}

/// True on a Metal-capable Apple platform — macOS, iOS, tvOS and
/// visionOS. All ship Metal, MetalPerformanceShaders and MPSGraph, so the
/// native `rlx-metal` backend builds and runs on each. **watchOS is
/// excluded**: it has no public Metal API, so the Metal backend never
/// builds there (Accelerate + CoreML remain available via
/// [`is_apple_platform`]).
pub const fn has_metal() -> bool {
    cfg!(all(target_vendor = "apple", not(target_os = "watchos")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicates_are_at_least_consistent() {
        // Consistency: Apple Silicon ⇒ aarch64 on an Apple platform, with
        // the AMX fast path. (Metal is *not* implied — watchOS is Apple
        // Silicon yet has no Metal.)
        if is_apple_silicon() {
            assert!(is_aarch64());
            assert!(is_apple_platform());
            assert!(has_amx());
        }
        // Every concrete Apple OS is an Apple platform.
        if is_macos() || is_ios() || is_tvos() || is_watchos() || is_visionos() {
            assert!(is_apple_platform());
        }
        assert_eq!(
            is_apple_platform(),
            is_macos() || is_ios() || is_tvos() || is_watchos() || is_visionos()
        );
        // Metal rides on every Apple platform *except* watchOS.
        assert_eq!(has_metal(), is_apple_platform() && !is_watchos());
        if has_metal() {
            assert!(is_apple_platform());
        }
        if is_watchos() {
            assert!(!has_metal());
        }
        // x86 can't be both.
        if is_x86_64() {
            assert!(!is_aarch64());
        }
        // NEON ⇒ aarch64 (today's predicate is exactly that).
        if has_neon() {
            assert!(is_aarch64());
        }
    }
}
