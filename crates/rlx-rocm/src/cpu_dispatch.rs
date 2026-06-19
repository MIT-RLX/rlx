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

//! Rust FFI bindings to the HIP-CPU validation path.
//!
//! Only compiled when `cargo build --features hip-cpu-validate`. The
//! corresponding C++ launch wrappers live in `cpp/cpu_dispatch.cpp`
//! (a one-line `#include` of rlx-cuda's wrapper layer), built by
//! `build.rs` against HIP-CPU's header-only runtime.
//!
//! This is a **dev-only** validation surface. Production HIP dispatch
//! goes through `crate::backend::RocmExecutable` + libloading + a real
//! AMD ROCm install. The CPU path lets us run the same `.cu` kernel
//! sources on CPU threads from Mac (or any host without an AMD driver)
//! so we can catch IR-lowering and kernel-logic bugs before paying for
//! cloud-GPU time.
//!
//! The FFI declarations are reused verbatim from `rlx-cuda` via
//! `#[path]` — same kernels, same wrappers, same C ABI. We just link
//! against rlx-rocm's own static lib (`rlx_rocm_cpu_dispatch.a`) which
//! happens to compile from the same TU as rlx-cuda's. Any binding
//! addition in rlx-cuda automatically appears here.

#![cfg(feature = "hip-cpu-validate")]

#[path = "../../rlx-cuda/src/cpu_dispatch.rs"]
mod shared;

pub use shared::*;
