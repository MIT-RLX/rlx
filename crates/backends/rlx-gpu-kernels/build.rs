// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Assembles the shared unary/activation CUDA/HIP kernel: the activation dispatch
// (`rlx_activation_apply`, op 0..28) is @generated from the single scalar-
// expression manifest in `rlxsl` — using the native hardware `erff` — then
// prepended to the hand-written kernel plumbing + cast selectors in
// `kernels/unary_main.cu`. The combined source is written to `$OUT_DIR/unary.cu`
// and `include_str!`d by `src/lib.rs`; it is JIT-compiled by NVRTC (CUDA) and
// hipRTC (ROCm), so it must use only device-portable intrinsics.

use std::{env, fs, path::Path};

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let main_src = fs::read_to_string("kernels/unary_main.cu").expect("read kernels/unary_main.cu");
    let activation = rlxsl::cuda_activation_module(rlxsl::OpcodeScheme::ReluFirst);
    let combined = format!("{activation}\n{main_src}");
    fs::write(Path::new(&out_dir).join("unary.cu"), combined).expect("write $OUT_DIR/unary.cu");

    // Backward: dx = act'(x)·dy. The derivative dispatch `rlx_activation_backward`
    // (op 0..17) is auto-differentiated from the forward manifest and prepended
    // to the plumbing (relu_backward + the switch kernel) in
    // `activation_backward_main.cu`.
    let bwd_main = fs::read_to_string("kernels/activation_backward_main.cu")
        .expect("read kernels/activation_backward_main.cu");
    let bwd = rlxsl::cuda_activation_backward_module();
    fs::write(
        Path::new(&out_dir).join("activation_backward.cu"),
        format!("{bwd}\n{bwd_main}"),
    )
    .expect("write $OUT_DIR/activation_backward.cu");

    // Standalone `binary` kernel: per-op math (`rlx_binary_apply`) @generated from
    // the shared rlxsl manifest (native `powf`), prepended to binary_main.cu.
    let bin_main =
        fs::read_to_string("kernels/binary_main.cu").expect("read kernels/binary_main.cu");
    let bin = rlxsl::binary::cuda_binary_module();
    fs::write(
        Path::new(&out_dir).join("binary.cu"),
        format!("{bin}\n{bin_main}"),
    )
    .expect("write $OUT_DIR/binary.cu");

    // Standalone `compare` kernel: per-op comparison (`rlx_compare_apply`)
    // @generated from the shared rlxsl manifest, prepended to compare_main.cu.
    let cmp_main =
        fs::read_to_string("kernels/compare_main.cu").expect("read kernels/compare_main.cu");
    let cmp = rlxsl::compare::cuda_compare_module();
    fs::write(
        Path::new(&out_dir).join("compare.cu"),
        format!("{cmp}\n{cmp_main}"),
    )
    .expect("write $OUT_DIR/compare.cu");

    println!("cargo:rerun-if-changed=kernels/unary_main.cu");
    println!("cargo:rerun-if-changed=kernels/activation_backward_main.cu");
    println!("cargo:rerun-if-changed=kernels/binary_main.cu");
    println!("cargo:rerun-if-changed=kernels/compare_main.cu");
    println!("cargo:rerun-if-changed=build.rs");
}
