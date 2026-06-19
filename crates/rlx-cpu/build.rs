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
    // The `blas` feature gates both the FFI extern in src/blas.rs and
    // the link directives below. `--no-default-features` skips both
    // and the kernels fall back to a portable scalar gemm.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BLAS");
    if std::env::var_os("CARGO_FEATURE_BLAS").is_none() {
        return;
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    // Cross-compiling wasm from macOS still reports host `cfg(target_os = "macos")`
    // in build.rs; skip native BLAS links for non-host targets.
    if target_arch == "wasm32" {
        return;
    }

    // macOS: vendored Accelerate framework provides cblas_sgemm.
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        return;
    }

    // Windows / Linux. `CARGO_FEATURE_BLAS_MKL` redirects the link from
    // OpenBLAS to Intel MKL (mkl_rt) — same `cblas_sgemm` ABI, but the
    // MKL dispatcher picks AVX-512 / VNNI kernels at runtime, which on
    // Windows hosts beats OpenBLAS by ~2x for the BERT-shape GEMMs the
    // MLM head emits. `mkl_rt` is the single-DLL "smart" router so the
    // build doesn't need to know whether the target is ILP64 or LP64 at
    // link time. Honours `MKL_ROOT` for non-system installs.
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_BLAS_MKL");
    let want_mkl = std::env::var_os("CARGO_FEATURE_BLAS_MKL").is_some();
    if want_mkl {
        println!("cargo:rerun-if-env-changed=MKL_ROOT");
        if let Ok(root) = std::env::var("MKL_ROOT") {
            // Intel's standard install lays the import lib under either
            // `lib/intel64` (oneAPI ≤ 2023) or directly under `lib`
            // (oneAPI ≥ 2024). Add both — extras are harmless.
            println!("cargo:rustc-link-search=native={root}/lib/intel64");
            println!("cargo:rustc-link-search=native={root}/lib");
            // Some Win installs co-locate the redistributable DLLs
            // under `redist`; surface them so a build artefact run from
            // the project root finds `mkl_rt.2.dll` without manual PATH
            // wrangling.
            println!("cargo:rustc-link-search=native={root}/redist/intel64");
        }
        println!("cargo:rustc-link-lib=mkl_rt");
        return;
    }

    // OpenBLAS path (default for non-macOS). Honour OPENBLAS_DIR
    // (matches openblas-src / burn-ndarray's blas-openblas-system
    // convention) and fall back to system search paths.
    println!("cargo:rerun-if-env-changed=OPENBLAS_DIR");
    println!("cargo:rerun-if-env-changed=OPENBLAS_LIB_DIR");
    if let Ok(dir) = std::env::var("OPENBLAS_LIB_DIR") {
        println!("cargo:rustc-link-search=native={dir}");
    } else if let Ok(root) = std::env::var("OPENBLAS_DIR") {
        println!("cargo:rustc-link-search=native={root}/lib");
    }
    // OpenBLAS provides cblas_sgemm under either name on different
    // distributions — try the explicit `libopenblas` first (Win
    // MSVC + most Linux distros), letting the linker fall back to
    // `cblas` if needed via the user's RUSTFLAGS.
    println!("cargo:rustc-link-lib=openblas");
}
