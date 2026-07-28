// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0
//
// Assembles the standalone unary/activation WGSL kernel: the activation
// dispatch (`rlx_activation_apply`) is @generated from the single scalar-
// expression manifest in `rlxsl`, then prepended to the hand-written
// kernel plumbing in `src/kernels/unary_main.wgsl`. The combined shader is
// written to `$OUT_DIR/unary.wgsl` and `include_str!`d by `kernels/mod.rs`.

use std::{env, fs, path::Path};

fn main() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let main_src = fs::read_to_string("src/kernels/unary_main.wgsl")
        .expect("read src/kernels/unary_main.wgsl");
    let activation = rlxsl::wgsl_activation_module(rlxsl::OpcodeScheme::ReluFirst);
    let combined = format!("{activation}\n{main_src}");
    fs::write(Path::new(&out_dir).join("unary.wgsl"), combined).expect("write $OUT_DIR/unary.wgsl");

    // Backward: dx = act'(x)·dy. Derivative dispatch auto-differentiated from the
    // forward manifest, prepended to the plumbing in activation_backward_main.wgsl.
    let bwd_main = fs::read_to_string("src/kernels/activation_backward_main.wgsl")
        .expect("read src/kernels/activation_backward_main.wgsl");
    let bwd = rlxsl::wgsl_activation_backward_module();
    fs::write(
        Path::new(&out_dir).join("activation_backward.wgsl"),
        format!("{bwd}\n{bwd_main}"),
    )
    .expect("write $OUT_DIR/activation_backward.wgsl");

    // Standalone `binary` kernel: per-op math (`rlx_binary_apply`) @generated from
    // the shared rlxsl manifest, prepended to the plumbing in binary_main.wgsl.
    let bin_main = fs::read_to_string("src/kernels/binary_main.wgsl")
        .expect("read src/kernels/binary_main.wgsl");
    let bin = rlxsl::binary::wgsl_binary_module();
    fs::write(
        Path::new(&out_dir).join("binary.wgsl"),
        format!("{bin}\n{bin_main}"),
    )
    .expect("write $OUT_DIR/binary.wgsl");

    // Standalone `compare` kernel: per-op comparison (`rlx_compare_apply`)
    // @generated from the shared rlxsl manifest, prepended to compare_main.wgsl.
    let cmp_main = fs::read_to_string("src/kernels/compare_main.wgsl")
        .expect("read src/kernels/compare_main.wgsl");
    let cmp = rlxsl::compare::wgsl_compare_module();
    fs::write(
        Path::new(&out_dir).join("compare.wgsl"),
        format!("{cmp}\n{cmp_main}"),
    )
    .expect("write $OUT_DIR/compare.wgsl");

    println!("cargo:rerun-if-changed=src/kernels/unary_main.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/activation_backward_main.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/binary_main.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/compare_main.wgsl");
    println!("cargo:rerun-if-changed=build.rs");
}
