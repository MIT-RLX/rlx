// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Export the generated warm-tier activation kernels as normal per-target source
// files, so they can be read / diffed / hand-checked exactly as the compiler
// sees them. This is the same code the backends compile at build time; it is
// NOT re-imported — the files are for inspection only.
//
//   cargo run -p rlxsl --example export            # -> ./rlxsl-generated/
//   cargo run -p rlxsl --example export /tmp/out   # -> /tmp/out/
//
// Each file uses the opcode scheme its backend actually dispatches with:
// relu-first for CUDA/wgpu, gelu-first for Vulkan (see rlx_ir::opcodes).

use rlxsl::OpcodeScheme::{GeluFirst, ReluFirst};
use std::{env, fs, path::PathBuf};

fn main() {
    let out = env::args().nth(1).unwrap_or_else(|| "rlxsl-generated".to_string());
    let dir = PathBuf::from(&out);
    fs::create_dir_all(&dir).expect("create output dir");

    // (filename, description, source)
    let artifacts: Vec<(&str, &str, String)> = vec![
        (
            "unary.wgsl",
            "WGSL (wgpu) — relu-first",
            rlxsl::wgsl_activation_module(ReluFirst),
        ),
        (
            "unary.cu",
            "CUDA / ROCm — relu-first, native erff",
            rlxsl::cuda_activation_module(ReluFirst),
        ),
        (
            "unary.glsl",
            "GLSL (native Vulkan) — gelu-first",
            rlxsl::glsl_activation_module(GeluFirst),
        ),
        (
            "unary.metal",
            "MSL (Metal switch form; runtime uses per-activation kernels) — relu-first",
            rlxsl::msl_activation_module(ReluFirst),
        ),
        // Backward: dx = d(activation)/dx · dy, auto-differentiated from the
        // forward manifest. Always relu-first (every backend's backward is).
        (
            "activation_backward.cu",
            "CUDA / ROCm backward — auto-differentiated",
            rlxsl::cuda_activation_backward_module(),
        ),
        (
            "activation_backward.wgsl",
            "WGSL backward — auto-differentiated",
            rlxsl::wgsl_activation_backward_module(),
        ),
        (
            "activation_backward.glsl",
            "GLSL backward — auto-differentiated",
            rlxsl::glsl_activation_backward_module(),
        ),
        (
            "activation_backward.metal",
            "MSL backward — auto-differentiated",
            rlxsl::msl_activation_backward_module(),
        ),
        // Double-single (2x f32 ≈ f64) extended-precision prelude — for
        // hardware with no f64 (e.g. Metal). Error-free transforms.
        (
            "double_single.wgsl",
            "double-single 2x f32 ≈ f64 — WGSL",
            rlxsl::dw::double_single_prelude(rlxsl::Lang::Wgsl),
        ),
        (
            "double_single.metal",
            "double-single 2x f32 ≈ f64 — MSL (Metal has no f64)",
            rlxsl::dw::double_single_prelude(rlxsl::Lang::Msl),
        ),
        (
            "double_single.cu",
            "double-single 2x f32 ≈ f64 — CUDA",
            rlxsl::dw::double_single_prelude(rlxsl::Lang::Cuda),
        ),
        (
            "unary.cl",
            "OpenCL-C (Intel oneAPI) — relu-first, native erf/rsqrt",
            rlxsl::opencl_activation_module(ReluFirst),
        ),
        (
            "activation_backward.cl",
            "OpenCL-C backward — auto-differentiated",
            rlxsl::opencl_activation_backward_module(),
        ),
    ];

    println!("rlxsl: exporting generated activation kernels to {}", dir.display());
    for (name, desc, src) in &artifacts {
        let path = dir.join(name);
        fs::write(&path, src).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
        println!("  {name:<26} {:>5} bytes  — {desc}", src.len());
    }
}
