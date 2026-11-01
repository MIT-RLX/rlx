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
    // The generated modules call the correctly-rounded helpers but do not
    // define them; each shader gets exactly one copy.
    let helpers = rlxsl::exact::wgsl_numeric_helpers();
    let activation = rlxsl::wgsl_activation_module(rlxsl::OpcodeScheme::ReluFirst);
    let combined = format!("{helpers}\n{activation}\n{main_src}");
    fs::write(Path::new(&out_dir).join("unary.wgsl"), combined).expect("write $OUT_DIR/unary.wgsl");

    // Backward: dx = act'(x)·dy. Derivative dispatch auto-differentiated from the
    // forward manifest, prepended to the plumbing in activation_backward_main.wgsl.
    let bwd_main = fs::read_to_string("src/kernels/activation_backward_main.wgsl")
        .expect("read src/kernels/activation_backward_main.wgsl");
    let bwd = rlxsl::wgsl_activation_backward_module();
    fs::write(
        Path::new(&out_dir).join("activation_backward.wgsl"),
        format!("{helpers}\n{bwd}\n{bwd_main}"),
    )
    .expect("write $OUT_DIR/activation_backward.wgsl");

    // Standalone `binary` kernel: per-op math (`rlx_binary_apply`) @generated from
    // the shared rlxsl manifest, prepended to the plumbing in binary_main.wgsl.
    let bin_main = fs::read_to_string("src/kernels/binary_main.wgsl")
        .expect("read src/kernels/binary_main.wgsl");
    let bin = rlxsl::binary::wgsl_binary_module();
    fs::write(
        Path::new(&out_dir).join("binary.wgsl"),
        format!("{helpers}\n{bin}\n{bin_main}"),
    )
    .expect("write $OUT_DIR/binary.wgsl");

    // Fused elementwise regions: binary and activation steps call the same
    // generated math as the standalone kernels, the activation switch keyed by
    // the region's own `sub` codes (`OpcodeScheme::Region`).
    let region_act = rlxsl::wgsl_activation_module(rlxsl::OpcodeScheme::Region);
    let region_main = fs::read_to_string("src/kernels/elementwise_region.wgsl")
        .expect("read src/kernels/elementwise_region.wgsl");
    fs::write(
        Path::new(&out_dir).join("elementwise_region.wgsl"),
        format!("{helpers}\n{bin}\n{region_act}\n{region_main}"),
    )
    .expect("write $OUT_DIR/elementwise_region.wgsl");

    // Batch norm: `inv = 1 / sqrt(var + eps)` with the correctly-rounded helpers.
    let bn_main = fs::read_to_string("src/kernels/batch_norm_inference.wgsl")
        .expect("read src/kernels/batch_norm_inference.wgsl");
    fs::write(
        Path::new(&out_dir).join("batch_norm_inference.wgsl"),
        format!("{helpers}\n{bn_main}"),
    )
    .expect("write $OUT_DIR/batch_norm_inference.wgsl");

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

    // conv3d and group-norm kernels: the store applies an optional folded activation with
    // the same generated math as the unary kernel.
    for k in ["conv3d", "conv3d_ctile", "group_norm"] {
        let src = fs::read_to_string(format!("src/kernels/{k}.wgsl"))
            .unwrap_or_else(|e| panic!("read src/kernels/{k}.wgsl: {e}"));
        // The output goes through `put` / `got`: into the arena binding, or
        // (`{k}_split.wgsl`) into binding 2, the output's own window — for an
        // arena striped past one storage binding, where input and output sit
        // in different stripes (`Conv3dParams::split`, `GroupNormParams::split`).
        let one =
            "fn put(i: u32, v: f32) { arena[i] = v; }\nfn got(i: u32) -> f32 { return arena[i]; }";
        let split = "@group(0) @binding(2) var<storage, read_write> dst: array<f32>;\n\
                     fn put(i: u32, v: f32) { dst[i] = v; }\nfn got(i: u32) -> f32 { return dst[i]; }";
        for (name, io) in [(k.to_string(), one), (format!("{k}_split"), split)] {
            if name.ends_with("_split") && k == "conv3d" {
                continue;
            }
            fs::write(
                Path::new(&out_dir).join(format!("{name}.wgsl")),
                format!("{helpers}\n{activation}\n{io}\n{src}"),
            )
            .unwrap_or_else(|e| panic!("write $OUT_DIR/{name}.wgsl: {e}"));
        }
        println!("cargo:rerun-if-changed=src/kernels/{k}.wgsl");
    }

    println!("cargo:rerun-if-changed=src/kernels/unary_main.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/activation_backward_main.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/binary_main.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/compare_main.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/elementwise_region.wgsl");
    println!("cargo:rerun-if-changed=src/kernels/batch_norm_inference.wgsl");
    println!("cargo:rerun-if-changed=build.rs");
}
