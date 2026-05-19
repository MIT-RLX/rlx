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

//! Smoke tests for the (currently scaffolded) ROCm backend.
//!
//! These exist to keep the scaffold honest:
//!   * The crate compiles on Mac, Linux, and CI runners.
//!   * `is_available()` returns false everywhere until real HIP
//!     bindings land — there's no false-positive risk.
//!   * Kernel sources reach through `include_str!` to rlx-cuda
//!     without breaking when those files move.
//!
//! Once real HIP dispatch is wired, replace the
//! `is_available_returns_false` test with the same structure
//! `rlx-cuda/tests/smoke.rs` uses (skip on hosts without driver,
//! dispatch + assert on hosts with one).

#[test]
fn is_available_returns_false_until_bindings_land() {
    assert!(
        !rlx_rocm::is_available(),
        "is_available() should be false until real HIP runtime \
         bindings replace the device.rs stub"
    );
}

#[test]
fn kernel_sources_are_reachable() {
    // Sanity: `include_str!` paths from rlx-rocm/src/kernels.rs
    // resolve. If rlx-cuda moves a kernel file, this test (and
    // the build) fail at compile-time, but assert non-empty here
    // anyway as a runtime tripwire.
    use rlx_rocm::kernels::*;
    assert!(!BINARY_CU.is_empty());
    assert!(!MATMUL_CU.is_empty());
    assert!(!ATTENTION_CU.is_empty());
    assert_eq!(KERNEL_COUNT, 33);
    assert!(!ELEMENTWISE_REGION_CU.is_empty());
}
