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

fn main() {
    // `rlx_cpu_blas` is set by THIS script exactly when a real BLAS is linked,
    // and gates the FFI + BLAS code paths in src/blas.rs / src/gguf_matmul.rs.
    // When unset the kernels use the portable SIMD/scalar gemm. Declaring it
    // keeps `#[cfg(rlx_cpu_blas)]` free of unexpected-cfg warnings even on the
    // targets where we deliberately don't set it (aarch64 Linux, wasm).
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_blas)");

    // The `blas` feature is the top-level switch; `--no-default-features`
    // (or a target with no BLAS) falls back to the portable gemm.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BLAS");
    if std::env::var_os("CARGO_FEATURE_BLAS").is_none() {
        return;
    }

    let target_vendor = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    // Cross-compiling wasm from macOS still reports host `cfg(target_os = "macos")`
    // in build.rs; no native BLAS there.
    if target_arch == "wasm32" {
        return;
    }

    // Every Apple platform (macOS, iOS, tvOS, watchOS, visionOS): the
    // Accelerate framework provides cblas_sgemm and the LAPACK symbols, and
    // routes GEMM through the AMX coprocessor on Apple Silicon — the fastest
    // CPU matmul path on every Apple device. Accelerate ships on all of them.
    if target_vendor == "apple" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=rlx_cpu_blas");
        return;
    }

    // Windows / Linux. `CARGO_FEATURE_BLAS_MKL` redirects the link from
    // OpenBLAS to Intel MKL (mkl_rt) — same `cblas_sgemm` ABI, but the MKL
    // dispatcher picks AVX-512 / VNNI kernels at runtime. `mkl_rt` is the
    // single-DLL "smart" router. Honours `MKL_ROOT` for non-system installs.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BLAS_MKL");
    if std::env::var_os("CARGO_FEATURE_BLAS_MKL").is_some() {
        println!("cargo:rerun-if-env-changed=MKL_ROOT");
        if let Ok(root) = std::env::var("MKL_ROOT") {
            println!("cargo:rustc-link-search=native={root}/lib/intel64");
            println!("cargo:rustc-link-search=native={root}/lib");
            println!("cargo:rustc-link-search=native={root}/redist/intel64");
        }
        println!("cargo:rustc-link-lib=mkl_rt");
        println!("cargo:rustc-cfg=rlx_cpu_blas");
        return;
    }

    // OpenBLAS path (default for non-Apple). Link it only where it actually
    // exists: x86_64 (Linux/Windows dev + CI ship libopenblas), or wherever the
    // user points us via OPENBLAS_LIB_DIR / OPENBLAS_DIR (e.g. an aarch64 board
    // that has libopenblas installed). On aarch64 Linux with no OpenBLAS — the
    // Raspberry Pi / cross-compile / QEMU case — skip the link and fall back to
    // the portable SIMD gemm so the crate still builds and runs.
    println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    let openblas_lib_dir = std::env::var("OPENBLAS_LIB_DIR").ok();
    let openblas_dir = std::env::var("OPENBLAS_DIR").ok();
    let pinned = openblas_lib_dir.is_some() || openblas_dir.is_some();
    if target_arch != "x86_64" && !pinned {
        println!(
            "cargo:warning=rlx-cpu: no OpenBLAS for {target_arch}-{target_os}; using the \
             portable SIMD gemm (set OPENBLAS_LIB_DIR to link one)"
        );
        return; // leave rlx_cpu_blas unset → SIMD fallback
    }
    if let Some(dir) = openblas_lib_dir {
        println!("cargo:rustc-link-search=native={dir}");
    } else if let Some(root) = openblas_dir {
        println!("cargo:rustc-link-search=native={root}/lib");
    }
    println!("cargo:rustc-link-lib=openblas");
    println!("cargo:rustc-cfg=rlx_cpu_blas");
}
