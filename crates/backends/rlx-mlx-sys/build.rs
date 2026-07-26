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

// Build script for rlx-mlx-sys.
//
// Two-stage build:
//   1. Drive MLX's CMake to produce a static libmlx.a (and the metal
//      kernel archive on macOS) inside OUT_DIR.
//   2. Compile our C++ shim (cpp/rlx_mlx_shim.cpp) and link it against
//      libmlx.a + the platform frameworks MLX itself depends on.
//
// On Linux, upstream MLX expects system BLAS/LAPACK headers (`lapacke.h`).
// When they are missing we bootstrap OpenBLAS into OUT_DIR (same spirit as
// MLX's Windows FetchContent OpenBLAS zip).

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const OPENBLAS_VERSION: &str = "0.3.31";

struct LapackPaths {
    include_dir: PathBuf,
    prefix_dir: PathBuf,
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mlx_src = manifest_dir.join("vendor").join("mlx");

    // MLX builds where `rlx-mlx` exposes a real backend (`rlx_mlx_host`):
    // macOS, Linux, Windows and iOS (device + simulator — MLX's CMake has a
    // native iOS branch and the Metal backend runs on-device / in the sim).
    // Every other target (tvOS / watchOS / visionOS / wasm / android) gets the
    // stub that links no MLX symbols, so skip the CMake cross-compile entirely.
    // Returning before the submodule check also means those builds don't
    // require `vendor/mlx` to be populated.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target = env::var("TARGET").unwrap_or_default();
    if !matches!(target_os.as_str(), "macos" | "linux" | "windows" | "ios") {
        return;
    }
    // iOS: arm64 device (aarch64-apple-ios) vs the arm64 simulator
    // (aarch64-apple-ios-sim). They differ only by SDK.
    let is_ios = target_os == "ios";
    let ios_sim = is_ios && target.ends_with("-sim");

    if !mlx_src.join("CMakeLists.txt").exists() {
        panic!(
            "rlx-mlx-sys: vendor/mlx is empty — run:\n\
             \n\
             \tgit submodule update --init rlx-mlx-sys/vendor/mlx\n\
             \n\
             Expected MLX source at {}",
            mlx_src.display()
        );
    }

    let is_macos = target_os == "macos";

    if is_macos
        && Command::new("xcrun")
            .args(["--find", "metal"])
            .output()
            .is_err()
    {
        eprintln!("warning: `xcrun metal` not found; MLX Metal kernels will fail to build");
    }

    // Stage 1: configure + build MLX into OUT_DIR.
    let build_cuda = linux_build_cuda(&target_os);
    let cmake_build_type = cmake_build_type_for_cargo_profile();
    println!("cargo:rerun-if-env-changed=RLX_MLX_CUDA");
    println!("cargo:rerun-if-env-changed=RLX_MLX_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=RLX_MLX_JOBS");

    let mut mlx_cfg = cmake::Config::new(&mlx_src);
    mlx_cfg
        .profile(cmake_build_type)
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("MLX_BUILD_PYTHON_STUBS", "OFF")
        .define(
            "MLX_BUILD_METAL",
            if is_macos || is_ios { "ON" } else { "OFF" },
        )
        .define("MLX_BUILD_CPU", "ON")
        .define("MLX_BUILD_GGUF", "OFF")
        .define("MLX_BUILD_SAFETENSORS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", cmake_build_type);

    // MLX pulls `fmt` in via CMake FetchContent (git clone + `git checkout
    // <tag>`). An interrupted prior configure can leave that tree half-checked
    // out (empty working tree, HEAD on the default branch); the next
    // `git checkout <tag>` then aborts with "local changes would be
    // overwritten" and fails every subsequent build. Seed a persistent,
    // correctly checked-out fmt once and hand CMake `FETCHCONTENT_SOURCE_DIR_FMT`
    // so no git runs at configure time — deterministic, offline-friendly after
    // the first fetch, and immune to the dirty-tree abort.
    seed_fmt_source(&mlx_src, &out_dir, &mut mlx_cfg);

    let macos_deploy = if is_macos {
        let deploy = env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".into());
        mlx_cfg.define("CMAKE_OSX_DEPLOYMENT_TARGET", deploy.as_str());
        mlx_cfg.env("CC", "/usr/bin/cc");
        mlx_cfg.env("CXX", "/usr/bin/c++");
        Some(deploy)
    } else {
        None
    };

    // iOS cross-compile: drive CMake into its `CMAKE_SYSTEM_NAME == iOS`
    // branch and point it at the device or simulator SDK. arm64 only (the
    // x86_64 sim is not an RLX target). MLX uses Accelerate for BLAS/LAPACK on
    // Apple, so no OpenBLAS bootstrap is needed.
    let ios_deploy = if is_ios {
        let sdk = if ios_sim {
            "iphonesimulator"
        } else {
            "iphoneos"
        };
        let deploy = env::var("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or_else(|_| "16.0".into());
        mlx_cfg
            .define("CMAKE_SYSTEM_NAME", "iOS")
            .define("CMAKE_OSX_SYSROOT", sdk)
            .define("CMAKE_OSX_ARCHITECTURES", "arm64")
            .define("CMAKE_OSX_DEPLOYMENT_TARGET", deploy.as_str());
        Some(deploy)
    } else {
        None
    };

    let use_ccache = (is_macos || is_ios)
        && env_flag("RLX_MLX_NO_CCACHE") != Some(true)
        && Command::new("ccache")
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
    mlx_cfg.define("MLX_USE_CCACHE", if use_ccache { "ON" } else { "OFF" });

    apply_cmake_parallelism(&mut mlx_cfg);

    if target_os == "linux" {
        let lapack = linux_lapack_paths(&out_dir).unwrap_or_else(|| {
            panic!(
                "rlx-mlx-sys: failed to locate or bootstrap BLAS/LAPACK on Linux.\n\
                 Install distro packages (fastest):\n\
                 \n\
                 \tsudo apt-get install libblas-dev liblapack-dev liblapacke-dev\n\
                 \n\
                 Or ensure `curl`, `tar`, and `cmake` are on PATH so build.rs can \
                 compile OpenBLAS into OUT_DIR."
            )
        });
        apply_lapack_hints(&mut mlx_cfg, &lapack);

        if build_cuda {
            eprintln!(
                "cargo:warning=rlx-mlx-sys: building MLX CUDA backend ({cmake_build_type}); \
                 first compile can take ~1h — use ccache + keep RLX_MLX_CUDA=1 stable"
            );
            mlx_cfg.define("MLX_BUILD_CUDA", "ON");
            if let Some(arch) = env::var("RLX_MLX_CUDA_ARCH").ok().filter(|s| !s.is_empty()) {
                mlx_cfg.define("MLX_CUDA_ARCHITECTURES", arch);
            }
            if let Some(cuda_root) = probe_linux_cuda_toolkit() {
                let root = cuda_root.to_string_lossy();
                mlx_cfg.define("CUDAToolkit_ROOT", root.as_ref());
                let bin = cuda_root.join("bin");
                if bin.is_dir() {
                    let path = env::var("PATH").unwrap_or_default();
                    mlx_cfg.env("PATH", format!("{}:{}", bin.display(), path));
                }
            }
        } else if probe_linux_cuda_toolkit().is_some() && probe_linux_cudnn().is_some() {
            eprintln!(
                "cargo:warning=rlx-mlx-sys: CUDA+cuDNN found but MLX CUDA backend skipped \
                 (CPU-only default). Set RLX_MLX_CUDA=1 or enable rlx-mlx-sys/cuda for GPU."
            );
        }
    }

    let mlx_build = mlx_cfg.build();

    let mlx_lib_dir = mlx_build.join("lib");
    let mlx_include_dir = mlx_build.join("include");

    println!("cargo:rustc-link-search=native={}", mlx_lib_dir.display());

    // Stage 2: compile the shim against MLX's installed headers.
    let mut shim = cc::Build::new();
    shim.cpp(true)
        .std("c++20")
        .file("cpp/rlx_mlx_shim.cpp")
        .include(&mlx_include_dir)
        .include(&mlx_src)
        .define("MLX_STATIC", None)
        .flag_if_supported("-fexceptions")
        .flag_if_supported("-fvisibility=hidden")
        .warnings(false);
    if let Some(ref deploy) = macos_deploy {
        shim.flag(format!("-mmacosx-version-min={deploy}"));
    }
    if let Some(ref deploy) = ios_deploy {
        // Match the deployment target MLX was built at, per SDK.
        shim.flag(if ios_sim {
            format!("-mios-simulator-version-min={deploy}")
        } else {
            format!("-miphoneos-version-min={deploy}")
        });
    }
    if target_os == "windows" {
        shim.define("NOMINMAX", None)
            .define("WIN32_LEAN_AND_MEAN", None)
            .define("NDEBUG", None)
            .flag("/MD");
    }
    shim.compile("rlx_mlx_shim");

    // Link mlx + platform frameworks. Order matters for static linking:
    // the shim references MLX symbols, MLX references frameworks.
    println!("cargo:rustc-link-lib=static=mlx");

    if is_macos {
        if let Some(deploy) = macos_deploy {
            // Match MLX CMake objects; avoid linking 14-built MLX at min 11.
            println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET={deploy}");
            println!("cargo:rustc-link-arg=-mmacosx-version-min={deploy}");
        }
        for fw in &["Metal", "Foundation", "QuartzCore", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        // JACCL (Thunderbolt RDMA + TCP distributed backend) builds as a
        // separate static lib when the macOS SDK is >= 26.2; libmlx.a
        // references its symbols. It installs next to libmlx.a (covered by
        // the link-search dir above). Its RDMA/Thunderbolt device path uses
        // IOKit + CoreFoundation. Link it only when actually present.
        if mlx_lib_dir.join("libjaccl.a").exists() {
            println!("cargo:rustc-link-lib=static=jaccl");
            println!("cargo:rustc-link-lib=framework=IOKit");
            println!("cargo:rustc-link-lib=framework=CoreFoundation");
        }
        // C++ runtime
        println!("cargo:rustc-link-lib=c++");
        // MLX Metal uses __builtin_available → ___isPlatformVersionAtLeast (compiler-rt).
        link_apple_clang_rt("osx");
    } else if is_ios {
        // iOS links the same Apple frameworks as macOS, minus the macOS-only
        // JACCL / IOKit Thunderbolt distributed path (not built on iOS).
        if let Some(deploy) = ios_deploy {
            println!("cargo:rustc-env=IPHONEOS_DEPLOYMENT_TARGET={deploy}");
        }
        for fw in &["Metal", "Foundation", "QuartzCore", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        println!("cargo:rustc-link-lib=c++");
        link_apple_clang_rt(if ios_sim { "iossim" } else { "ios" });
    } else if target_os == "linux" {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
        if build_cuda {
            if let Some(cuda_root) = probe_linux_cuda_toolkit() {
                link_linux_cuda_libs(&cuda_root);
            }
        }
    } else if target_os == "windows" {
        link_windows_mlx_deps(&out_dir);
    }

    // Re-run if the shim or vendored MLX commit changes.
    println!("cargo:rerun-if-changed=cpp/rlx_mlx_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/rlx_mlx_shim.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=vendor/mlx/mlx/version.h");
}

/// Pre-seed the `fmt` source MLX's CMake fetches so no `git clone`/`git
/// checkout` runs during configure (the step that fails with "Failed to
/// checkout tag" when a prior populate was interrupted). Clones the pinned tag
/// once into a persistent cache shared across every rlx-mlx-sys build variant
/// of this profile, then points CMake at it via `FETCHCONTENT_SOURCE_DIR_FMT`.
///
/// Falls back to the previous behaviour — drop any half-populated fmt tree so
/// FetchContent re-clones from scratch — when the tag can't be determined or
/// git isn't available (e.g. offline with no cache yet), so the build still
/// works exactly as before in that case.
fn seed_fmt_source(mlx_src: &Path, out_dir: &Path, cfg: &mut cmake::Config) {
    let seeded = fmt_git_tag(mlx_src).and_then(|tag| {
        // OUT_DIR is `.../target/<profile>/build/<pkg>-<hash>/out`; its
        // grandparent `.../build` is shared by all rlx-mlx-sys variants, so a
        // cache there is fetched once and reused everywhere.
        let build_root = out_dir.ancestors().nth(2)?;
        ensure_fmt_cache(build_root, &tag)
    });

    if let Some(cache) = seeded {
        cfg.define(
            "FETCHCONTENT_SOURCE_DIR_FMT",
            cache.to_string_lossy().as_ref(),
        );
        // Re-seed if MLX bumps the pinned fmt tag.
        println!("cargo:rerun-if-changed=vendor/mlx/CMakeLists.txt");
        return;
    }

    // Fallback: couldn't seed. Remove any dirty/half-populated fmt populate
    // dirs so FetchContent's own clone starts clean instead of aborting.
    let deps = out_dir.join("build").join("_deps");
    for name in ["fmt-src", "fmt-subbuild", "fmt-build"] {
        force_remove_dir_all(&deps.join(name));
    }
}

/// The `GIT_TAG` pinned for the fmt FetchContent dependency in MLX's top-level
/// CMakeLists.txt, so the seeded checkout always matches the vendored MLX.
fn fmt_git_tag(mlx_src: &Path) -> Option<String> {
    let text = std::fs::read_to_string(mlx_src.join("CMakeLists.txt")).ok()?;
    let mut saw_fmt_repo = false;
    for line in text.lines() {
        let l = line.trim();
        if l.contains("fmtlib/fmt.git") {
            saw_fmt_repo = true;
            continue;
        }
        if saw_fmt_repo {
            if let Some(rest) = l.strip_prefix("GIT_TAG") {
                let tag = rest.trim().trim_matches('"').trim().to_string();
                if !tag.is_empty() {
                    return Some(tag);
                }
            }
            // GIT_TAG follows GIT_REPOSITORY inside the same Declare block; stop
            // if we reach its end without finding one.
            if l.ends_with(')') || l.contains("FetchContent_MakeAvailable") {
                break;
            }
        }
    }
    None
}

/// Ensure a full fmt checkout at `tag` exists under `build_root`, returning its
/// path. Clones into a unique temp dir then atomically renames it into place so
/// concurrent build variants racing to seed the shared cache can't corrupt it.
fn ensure_fmt_cache(build_root: &Path, tag: &str) -> Option<PathBuf> {
    let cache = build_root.join(format!("rlx-mlx-sys-fmt-{tag}"));
    if fmt_checkout_ok(&cache) {
        return Some(cache);
    }
    std::fs::create_dir_all(build_root).ok()?;

    let tmp = build_root.join(format!("rlx-mlx-sys-fmt-{tag}.tmp.{}", std::process::id()));
    force_remove_dir_all(&tmp);
    let cloned = Command::new("git")
        .args([
            "clone",
            "--depth",
            "1",
            "--branch",
            tag,
            "--config",
            "advice.detachedHead=false",
            "https://github.com/fmtlib/fmt.git",
        ])
        .arg(&tmp)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    if !cloned || !fmt_checkout_ok(&tmp) {
        force_remove_dir_all(&tmp);
        // A concurrent build may have finished the cache while we cloned.
        return fmt_checkout_ok(&cache).then_some(cache);
    }

    match std::fs::rename(&tmp, &cache) {
        Ok(()) => Some(cache),
        Err(_) => {
            // Lost the publish race (or cross-dir rename); a valid cache wins.
            force_remove_dir_all(&tmp);
            fmt_checkout_ok(&cache).then_some(cache)
        }
    }
}

/// A directory is a usable fmt source if it has fmt's CMakeLists and headers.
fn fmt_checkout_ok(dir: &Path) -> bool {
    dir.join("CMakeLists.txt").exists() && dir.join("include/fmt/format.h").exists()
}

/// Remove a directory tree robustly: retry after clearing read-only bits (git
/// pack files are read-only) and, as a last resort, shell out to the platform
/// force-remove. The plain `remove_dir_all` this replaces silently swallowed
/// partial failures, which is how half-deleted fmt trees leaked out.
fn force_remove_dir_all(path: &Path) {
    if !path.exists() {
        return;
    }
    if std::fs::remove_dir_all(path).is_ok() {
        return;
    }
    fn make_writable(dir: &Path) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            let Ok(meta) = std::fs::symlink_metadata(&p) else {
                continue;
            };
            let mut perms = meta.permissions();
            if perms.readonly() {
                #[allow(clippy::permissions_set_readonly_false)]
                perms.set_readonly(false);
                let _ = std::fs::set_permissions(&p, perms);
            }
            if meta.file_type().is_dir() {
                make_writable(&p);
            }
        }
    }
    make_writable(path);
    if std::fs::remove_dir_all(path).is_ok() {
        return;
    }
    #[cfg(windows)]
    let _ = Command::new("cmd")
        .args(["/C", "rmdir", "/S", "/Q"])
        .arg(path)
        .status();
    #[cfg(not(windows))]
    let _ = Command::new("rm").arg("-rf").arg(path).status();
}

/// `libclang_rt.<variant>` — required for `___isPlatformVersionAtLeast` from
/// MLX `__builtin_available` checks. Rust's default link line omits it
/// (`-nodefaultlibs`). `variant` is the Apple runtime suffix: `"osx"` (macOS),
/// `"ios"` (device) or `"iossim"` (simulator).
fn link_apple_clang_rt(variant: &str) {
    let libname = format!("libclang_rt.{variant}.a");
    let output = match Command::new("clang")
        .arg(format!("--print-file-name={libname}"))
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => {
            eprintln!(
                "cargo:warning=rlx-mlx-sys: could not run `clang --print-file-name={libname}`"
            );
            return;
        }
    };
    let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if path.is_empty() || path == libname {
        eprintln!("cargo:warning=rlx-mlx-sys: clang did not resolve {libname}");
        return;
    }
    let path = PathBuf::from(path);
    if !path.is_file() {
        eprintln!(
            "cargo:warning=rlx-mlx-sys: {libname} not found at {}",
            path.display()
        );
        return;
    }
    if variant == "osx" {
        // Proven path on macOS: search dir + static lib.
        if let Some(parent) = path.parent() {
            println!("cargo:rustc-link-search=native={}", parent.display());
        }
        println!("cargo:rustc-link-lib=static=clang_rt.{variant}");
    } else {
        // iOS / simulator clang_rt archives are universal (fat) Mach-O. rustc's
        // own static-archive reader rejects those ("Unsupported archive
        // identifier"), so hand the full path straight to the linker (ld64),
        // which is fat-aware and slices the right arch.
        println!("cargo:rustc-link-arg={}", path.display());
    }
}

fn linux_build_cuda(target_os: &str) -> bool {
    if target_os != "linux" {
        return false;
    }
    match env_flag("RLX_MLX_CUDA") {
        Some(false) => return false,
        Some(true) => {}
        None if env::var("CARGO_FEATURE_CUDA").is_ok() => {}
        None => return false,
    }
    probe_linux_cuda_toolkit().is_some() && probe_linux_cudnn().is_some()
}

/// Match Cargo profile on Linux so `cargo build` (debug) skips `-O3` nvcc compiles.
/// macOS/Windows stay on Release — Metal/MSVC paths are already cached and
/// Debug MLX there buys little.
fn cmake_build_type_for_cargo_profile() -> &'static str {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "linux" {
        return "Release";
    }
    match env::var("PROFILE").as_deref() {
        Ok("debug") => "Debug",
        Ok("release") => "Release",
        _ => "RelWithDebInfo",
    }
}

fn apply_cmake_parallelism(cfg: &mut cmake::Config) {
    if let Ok(jobs) = env::var("RLX_MLX_JOBS") {
        if !jobs.is_empty() {
            cfg.env("CMAKE_BUILD_PARALLEL_LEVEL", jobs);
        }
    }
}

fn env_flag(name: &str) -> Option<bool> {
    let v = env::var(name).ok()?;
    match v.trim().to_ascii_lowercase().as_str() {
        "0" | "off" | "false" | "no" => Some(false),
        "1" | "on" | "true" | "yes" => Some(true),
        _ => None,
    }
}

fn probe_linux_cuda_toolkit() -> Option<PathBuf> {
    for root in [
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-12.6",
    ] {
        let path = PathBuf::from(root);
        let lib64 = path.join("lib64");
        if lib64.join("libcudart.so").exists() || lib64.join("libcudart.so.12").exists() {
            return Some(path);
        }
    }
    None
}

fn probe_linux_cudnn() -> Option<PathBuf> {
    for include in ["/usr/include/x86_64-linux-gnu", "/usr/include"] {
        let path = PathBuf::from(include);
        if path.join("cudnn.h").exists() {
            return Some(path);
        }
    }
    None
}

fn link_linux_cuda_libs(cuda_root: &Path) {
    let lib_dir = cuda_root.join("lib64");
    let stubs = lib_dir.join("stubs");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    if stubs.is_dir() {
        println!("cargo:rustc-link-search=native={}", stubs.display());
    }
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    for lib in [
        "cudart",
        "cublas",
        "cublasLt",
        "cufft",
        "nvrtc",
        "cuda",
        "cudnn",
        "cudnn_graph",
        "cudnn_engines_runtime_compiled",
        "cudnn_ops",
        "cudnn_cnn",
        "cudnn_adv",
        "cudnn_heuristic",
        "cudnn_engines_precompiled",
    ] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn link_windows_mlx_deps(out_dir: &Path) {
    println!("cargo:rustc-link-lib=advapi32");
    link_lib_from_out_dir(out_dir, "dl.lib", "dl");
    link_lib_from_out_dir(out_dir, "libopenblas.lib", "libopenblas");
}

fn link_lib_from_out_dir(out_dir: &Path, file_name: &str, link_name: &str) {
    if let Some(path) = find_file_under(out_dir, file_name, 8) {
        if let Some(dir) = path.parent() {
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib={link_name}");
            return;
        }
    }
    eprintln!(
        "cargo:warning=rlx-mlx-sys: {file_name} not found under OUT_DIR; \
         MLX Windows link may fail"
    );
}

fn find_file_under(root: &Path, file_name: &str, max_depth: usize) -> Option<PathBuf> {
    fn walk(dir: &Path, file_name: &str, depth: usize, max_depth: usize) -> Option<PathBuf> {
        if depth > max_depth {
            return None;
        }
        let entries = std::fs::read_dir(dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.file_name().and_then(|n| n.to_str()) == Some(file_name) {
                return Some(path);
            }
            if path.is_dir() {
                if let Some(found) = walk(&path, file_name, depth + 1, max_depth) {
                    return Some(found);
                }
            }
        }
        None
    }
    walk(root, file_name, 0, max_depth)
}

fn apply_lapack_hints(cfg: &mut cmake::Config, lapack: &LapackPaths) {
    let include = lapack.include_dir.to_string_lossy();
    cfg.define("LAPACK_INCLUDE_DIRS", include.as_ref());
    cfg.define("BLAS_INCLUDE_DIRS", include.as_ref());
    cfg.define("BLA_VENDOR", "OpenBLAS");
    cfg.env("BLAS_HOME", &lapack.prefix_dir);
    cfg.env("LAPACK_ROOT", &lapack.prefix_dir);
    cfg.env(
        "CMAKE_PREFIX_PATH",
        lapack.prefix_dir.to_string_lossy().into_owned(),
    );

    let lib = lapack.prefix_dir.join("lib/libopenblas.a");
    if lib.exists() {
        let lib = lib.to_string_lossy();
        cfg.define("LAPACK_LIBRARIES", lib.as_ref());
        cfg.define("BLAS_LIBRARIES", lib.as_ref());
    }
}

fn linux_lapack_paths(out_dir: &Path) -> Option<LapackPaths> {
    if let Some(paths) = probe_system_lapack() {
        return Some(paths);
    }

    eprintln!(
        "cargo:warning=rlx-mlx-sys: lapacke.h not found; building OpenBLAS v{OPENBLAS_VERSION} into OUT_DIR (first build may take several minutes)"
    );
    bootstrap_openblas(out_dir)
}

fn probe_system_lapack() -> Option<LapackPaths> {
    for include_dir in [
        "/usr/include/x86_64-linux-gnu",
        "/usr/include",
        "/usr/local/include",
        "/usr/include/openblas",
        "/usr/local/include/openblas",
    ] {
        let include = PathBuf::from(include_dir);
        if include.join("lapacke.h").exists() {
            let prefix_dir = if include_dir.contains("/openblas") {
                include.parent().unwrap_or(Path::new("/usr")).to_path_buf()
            } else if Path::new("/usr/local").join("include/lapacke.h").exists()
                && include_dir.starts_with("/usr/local")
            {
                PathBuf::from("/usr/local")
            } else {
                PathBuf::from("/usr")
            };
            return Some(LapackPaths {
                include_dir: include,
                prefix_dir,
            });
        }
    }
    None
}

fn bootstrap_openblas(out_dir: &Path) -> Option<LapackPaths> {
    let tarball = out_dir.join(format!("OpenBLAS-{OPENBLAS_VERSION}.tar.gz"));
    let src_dir = out_dir.join(format!("OpenBLAS-{OPENBLAS_VERSION}"));
    let url = format!(
        "https://github.com/OpenMathLib/OpenBLAS/releases/download/v{OPENBLAS_VERSION}/OpenBLAS-{OPENBLAS_VERSION}.tar.gz"
    );

    if !src_dir.join("CMakeLists.txt").exists() {
        if !tarball.exists() {
            download_file(&url, &tarball)?;
        }
        extract_tar_gz(&tarball, out_dir)?;
    }

    let install = cmake::Config::new(&src_dir)
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("NOFORTRAN", "ON")
        .define("C_LAPACK", "ON")
        .define("DYNAMIC_ARCH", "OFF")
        .build();

    let include_dir = lapack_include_dir(&install)?;
    Some(LapackPaths {
        include_dir,
        prefix_dir: install,
    })
}

fn lapack_include_dir(prefix: &Path) -> Option<PathBuf> {
    for include_dir in [prefix.join("include/openblas"), prefix.join("include")] {
        if include_dir.join("lapacke.h").exists() {
            return Some(include_dir);
        }
    }
    eprintln!(
        "cargo:warning=rlx-mlx-sys: OpenBLAS install missing lapacke.h under {}",
        prefix.join("include").display()
    );
    None
}

fn download_file(url: &str, dest: &Path) -> Option<()> {
    if dest.exists() {
        return Some(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).ok()?;
    }

    if Command::new("curl").arg("--version").output().is_ok() {
        let status = Command::new("curl")
            .args(["-fsSL", "-o"])
            .arg(dest)
            .arg(url)
            .status()
            .ok()?;
        if status.success() {
            return Some(());
        }
    }

    if Command::new("wget").arg("--version").output().is_ok() {
        let status = Command::new("wget")
            .arg("-O")
            .arg(dest)
            .arg(url)
            .status()
            .ok()?;
        if status.success() {
            return Some(());
        }
    }

    None
}

fn extract_tar_gz(archive: &Path, dest_dir: &Path) -> Option<()> {
    let status = Command::new("tar")
        .arg("-xzf")
        .arg(archive)
        .arg("-C")
        .arg(dest_dir)
        .status()
        .ok()?;
    if status.success() { Some(()) } else { None }
}
