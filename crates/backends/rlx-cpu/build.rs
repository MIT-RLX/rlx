// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

fn main() {
    // `rlx_cpu_blas` is set by THIS script exactly when a real BLAS is linked,
    // and gates the FFI + BLAS code paths in src/blas.rs / src/gguf_matmul.rs.
    // When unset the kernels use the portable SIMD/scalar gemm. Declaring it
    // keeps `#[cfg(rlx_cpu_blas)]` free of unexpected-cfg warnings even on the
    // targets where we deliberately don't set it (aarch64 Linux, wasm).
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_blas)");
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_blas_accelerate)");
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_blas_openblas)");
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_blas_mkl)");

    // Apple AMX / SME matrix-coprocessor fast paths (see Cargo.toml). Each
    // `amx-*` cargo feature lights the matching `rlx_cpu_amx_*` cfg — but ONLY
    // on `target_vendor = "apple"`, since the underlying hardware (AMX / SME)
    // exists nowhere else. Enabling the feature on a non-Apple target compiles
    // cleanly and the path is simply never taken (the cfg stays unset). Runtime
    // sysctl probes gate the actual dispatch on top of these compile cfgs.
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_amx_bnns)");
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_amx_dense)");
    println!("cargo:rustc-check-cfg=cfg(rlx_cpu_amx_sme)");
    {
        let is_apple = std::env::var("CARGO_CFG_TARGET_VENDOR").unwrap_or_default() == "apple";
        for (feat, cfg) in [
            ("CARGO_FEATURE_AMX_BNNS", "rlx_cpu_amx_bnns"),
            ("CARGO_FEATURE_AMX_DENSE", "rlx_cpu_amx_dense"),
            ("CARGO_FEATURE_AMX_SME", "rlx_cpu_amx_sme"),
        ] {
            println!("cargo:rerun-if-env-changed={feat}");
            if is_apple && std::env::var_os(feat).is_some() {
                println!("cargo:rustc-cfg={cfg}");
            }
        }
    }

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
        println!("cargo:rustc-cfg=rlx_cpu_blas_accelerate");
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
        println!("cargo:rustc-cfg=rlx_cpu_blas_mkl");
        return;
    }

    // OpenBLAS path (default for non-Apple). Link it only where it actually
    // exists: x86_64 Linux (dev + CI ship libopenblas), or wherever the user
    // points us via OPENBLAS_LIB_DIR / OPENBLAS_DIR. On aarch64 Linux with no
    // OpenBLAS — and on fresh Windows boxes without an OpenBLAS install —
    // skip the link and fall back to the portable SIMD gemm so the crate
    // still builds.
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
    // Windows x86_64: only link when an import lib is findable. Fresh VMs
    // and CI images often lack OpenBLAS; LNK1181 is worse than the portable
    // gemm fallback. Official OpenBLAS MSVC zips ship `libopenblas.lib`
    // (not `openblas.lib`).
    let win_lib = if target_os == "windows" {
        match find_windows_openblas_lib(openblas_lib_dir.as_deref(), openblas_dir.as_deref()) {
            Some(hit) => Some(hit),
            None if pinned => None, // user pinned a dir; still try generic link below
            None => {
                println!(
                    "cargo:warning=rlx-cpu: openblas.lib/libopenblas.lib not found; using the \
                     portable SIMD gemm (set OPENBLAS_LIB_DIR or install OpenBLAS under \
                     C:\\OpenBLAS)"
                );
                return;
            }
        }
    } else {
        None
    };

    if let Some((lib_dir, link_name)) = win_lib {
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-lib={link_name}");
    } else if let Some(dir) = openblas_lib_dir {
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=openblas");
    } else if let Some(root) = openblas_dir {
        println!("cargo:rustc-link-search=native={root}/lib");
        println!("cargo:rustc-link-lib=openblas");
    } else {
        println!("cargo:rustc-link-lib=openblas");
    }
    println!("cargo:rustc-cfg=rlx_cpu_blas");
    println!("cargo:rustc-cfg=rlx_cpu_blas_openblas");
}

/// `(lib_dir, rustc_link_lib_name)` for a Windows OpenBLAS install.
fn find_windows_openblas_lib(
    openblas_lib_dir: Option<&str>,
    openblas_dir: Option<&str>,
) -> Option<(String, &'static str)> {
    use std::path::PathBuf;
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(dir) = openblas_lib_dir {
        dirs.push(PathBuf::from(dir));
    }
    if let Some(root) = openblas_dir {
        dirs.push(PathBuf::from(root).join("lib"));
    }
    if let Ok(lib) = std::env::var("LIB") {
        dirs.extend(lib.split(';').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    for cand in [
        r"C:\OpenBLAS\lib",
        r"C:\openblas\lib",
        r"C:\vcpkg\installed\x64-windows\lib",
        r"C:\Program Files\OpenBLAS\lib",
    ] {
        dirs.push(PathBuf::from(cand));
    }
    for d in dirs {
        // Prefer the official MSVC zip name, then the bare `openblas.lib`.
        if d.join("libopenblas.lib").is_file() {
            return Some((d.to_string_lossy().into_owned(), "libopenblas"));
        }
        if d.join("openblas.lib").is_file() {
            return Some((d.to_string_lossy().into_owned(), "openblas"));
        }
    }
    None
}
