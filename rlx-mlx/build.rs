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

// Build script for rlx-mlx.
//
// Two-stage build:
//   1. Drive MLX's CMake to produce a static libmlx.a (and the metal
//      kernel archive on macOS) inside OUT_DIR.
//   2. Compile our C++ shim (cpp/rlx_mlx_shim.cpp) and link it against
//      libmlx.a + the platform frameworks MLX itself depends on.
//
// On macOS, MLX builds its Metal kernels by default; that requires
// `xcrun metal` / `xcrun metallib` on PATH. The `xcrun --find metal`
// check below produces a clearer error than the deep-in-cmake one if
// the toolchain is missing.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mlx_src = manifest_dir.parent().unwrap().join("vendor").join("mlx");

    if !mlx_src.join("CMakeLists.txt").exists() {
        panic!(
            "vendor/mlx is empty — did you `git submodule update --init` after \
             cloning? Expected MLX source at {}",
            mlx_src.display()
        );
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
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
    let mlx_build = cmake::Config::new(&mlx_src)
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        .define("MLX_BUILD_PYTHON_BINDINGS", "OFF")
        .define("MLX_BUILD_PYTHON_STUBS", "OFF")
        .define("MLX_BUILD_METAL", if is_macos { "ON" } else { "OFF" })
        .define("MLX_BUILD_CPU", "ON")
        .define("MLX_BUILD_GGUF", "OFF")
        .define("MLX_BUILD_SAFETENSORS", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build();

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
        .flag_if_supported("-fexceptions")
        .flag_if_supported("-fvisibility=hidden")
        .warnings(false);
    shim.compile("rlx_mlx_shim");

    // Link mlx + platform frameworks. Order matters for static linking:
    // the shim references MLX symbols, MLX references frameworks.
    println!("cargo:rustc-link-lib=static=mlx");

    if is_macos {
        for fw in &["Metal", "Foundation", "QuartzCore", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        // C++ runtime
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Re-run if the shim or vendored MLX commit changes.
    println!("cargo:rerun-if-changed=cpp/rlx_mlx_shim.cpp");
    println!("cargo:rerun-if-changed=cpp/rlx_mlx_shim.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../vendor/mlx/mlx/version.h");
}
