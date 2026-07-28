// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! rlx-cuda build script.
//!
//! By default this is a no-op — the crate is pure Rust + cudarc and
//! NVRTC-compiles its `.cu` sources at runtime.
//!
//! With `--features hip-cpu-validate`, we compile a single C++ TU
//! (`cpp/cpu_dispatch.cpp`) against HIP-CPU's header-only runtime,
//! producing a static lib that the Rust crate links against. Use only inside
//! the linux-gnu Docker image (`just test-hip-cpu-validate`); do not enable on
//! macOS hosts.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(feature = "hip-cpu-validate")]
    build_hip_cpu();
}

#[cfg(feature = "hip-cpu-validate")]
fn build_hip_cpu() {
    use std::path::Path;
    let hip_cpu_include = Path::new("docker/vendor/HIP-CPU/include");
    if !hip_cpu_include.exists() {
        panic!(
            "rlx-cuda hip-cpu-validate: missing HIP-CPU headers at {}\n\
             HIP-CPU is fetched only inside Docker (linux-gnu libstdc++).\n\
             \n\
             \tjust test-hip-cpu-validate\n\
             \n\
             (clones into rlx-cuda/docker/vendor/HIP-CPU via rlx-cuda/docker/fetch-hip-cpu.sh)",
            hip_cpu_include.display()
        );
    }

    let kernels_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../rlx-gpu-kernels/kernels");

    println!("cargo:rerun-if-changed=cpp/cpu_dispatch.cpp");
    println!("cargo:rerun-if-changed={}", kernels_dir.display());

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("cpp/cpu_dispatch.cpp")
        .include(hip_cpu_include)
        .include(&kernels_dir)
        // HIP-CPU runtime mode — selects the CPU thread-pool backend
        // instead of any GPU runtime.
        .define("__HIP_CPU_RT__", None)
        // Allow `__global__`, `__device__`, etc. attributes used in
        // our `.cu` files to be treated as no-ops on the CPU side.
        .flag_if_supported("-Wno-unknown-attributes")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("rlx_cuda_cpu_dispatch");

    // pthread for HIP-CPU's std::thread fallback.
    println!("cargo:rustc-link-lib=pthread");
}
