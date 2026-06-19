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

/// True on Apple Silicon (M-series) Macs.
pub const fn is_apple_silicon() -> bool {
    cfg!(all(target_arch = "aarch64", target_os = "macos"))
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

/// True if AVX2 is reasonably likely to be present at runtime.
/// This is a *static* prediction — explicit runtime detection
/// belongs in the dispatch path; this is for "should we even
/// compile the AVX2 code path?"
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

/// True on the Metal-capable platform (macOS, plus iOS/tvOS in the
/// future if RLX adds those targets).
pub const fn has_metal() -> bool {
    cfg!(target_os = "macos")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicates_are_at_least_consistent() {
        // Consistency: aarch64-on-macos implies apple silicon.
        if is_apple_silicon() {
            assert!(is_aarch64());
            assert!(is_macos());
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
