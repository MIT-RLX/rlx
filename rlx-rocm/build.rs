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

//! rlx-rocm build script.
//!
//! By default this is a no-op — the crate is pure Rust + libloading and
//! hipRTC-compiles its `.cu` sources at runtime against a real ROCm
//! install.
//!
//! With `--features hip-cpu-validate`, we compile a single C++ TU
//! (`cpp/cpu_dispatch.cpp`, which #includes rlx-cuda's wrapper layer)
//! against HIP-CPU's header-only runtime, producing a static lib that
//! the Rust crate links against. The CPU path is *strictly* a
//! development convenience — it lets us run the same kernel sources
//! on CPU threads on Mac/Docker for parity-check purposes, without
//! renting an AMD GPU.
//!
//! The HIP-CPU submodule lives in `rlx-cuda/vendor/HIP-CPU` (shared
//! with rlx-cuda's harness — same upstream, same kernel sources, no
//! point in two copies of the headers).

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "hip-cpu-validate")]
    build_hip_cpu();
}

#[cfg(feature = "hip-cpu-validate")]
fn build_hip_cpu() {
    use std::path::Path;
    let hip_cpu_include = Path::new("../rlx-cuda/vendor/HIP-CPU/include");
    if !hip_cpu_include.exists() {
        panic!(
            "rlx-rocm hip-cpu-validate: missing HIP-CPU headers at {}\n\
             Initialize the submodule (shared with rlx-cuda) with:\n\
             \n\
             \tgit submodule add https://github.com/ROCm-Developer-Tools/HIP-CPU.git \\\n\
             \t    rlx-cuda/vendor/HIP-CPU\n\
             \tgit submodule update --init\n\
             \n\
             (or whatever upstream HIP-CPU mirror is current)",
            hip_cpu_include.display()
        );
    }

    println!("cargo:rerun-if-changed=cpp/cpu_dispatch.cpp");
    println!("cargo:rerun-if-changed=../rlx-cuda/cpp/cpu_dispatch.cpp");
    println!("cargo:rerun-if-changed=../rlx-cuda/src/kernels");

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/cpu_dispatch.cpp")
        .include(hip_cpu_include)
        .include("../rlx-cuda/src/kernels")
        // HIP-CPU runtime mode — selects the CPU thread-pool backend
        // instead of any GPU runtime.
        .define("__HIP_CPU_RT__", None)
        // Allow `__global__`, `__device__`, etc. attributes used in
        // our `.cu` files to be treated as no-ops on the CPU side.
        .flag_if_supported("-Wno-unknown-attributes")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("rlx_rocm_cpu_dispatch");

    // pthread for HIP-CPU's std::thread fallback.
    println!("cargo:rustc-link-lib=pthread");
}
