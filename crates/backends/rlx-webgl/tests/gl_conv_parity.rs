// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The WebGL2 conv3d shaders themselves, in a browser, against the plan's
//! CPU executor (which the native tests hold to rlx-cpu): the multi-channel
//! kernel (`fs_conv3d_quads` with 1–4 targets, `FS_PACK_W4`,
//! `fs_unpack_quads`) on strided, anisotropic, dilated and padded convs,
//! batches, channel counts that leave the last quad or pass short and need
//! several passes, and the one-channel kernel it falls back to (grouped, fewer
//! than four channels).
//!
//! ```sh
//! WASM_BINDGEN_USE_BROWSER=1 NO_HEADLESS=1 \
//!   CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
//!   cargo test -p rlx-webgl --target wasm32-unknown-unknown --test gl_conv_parity
//! # then open the printed http://127.0.0.1:8000 page in a browser with WebGL2
//! ```
#![cfg(target_arch = "wasm32")]

use rlx_ir::{DType, Graph, Shape};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

struct Conv {
    n: usize,
    cin: usize,
    cout: usize,
    dims: [usize; 3],
    k: [usize; 3],
    s: [usize; 3],
    p: [usize; 3],
    d: [usize; 3],
    groups: usize,
}

fn run(c: &Conv) {
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let xn = c.n * c.cin * c.dims.iter().product::<usize>();
    let wn = c.cout * (c.cin / c.groups) * c.k.iter().product::<usize>();
    let xs: Vec<f32> = (0..xn).map(|i| (i as f32 * 0.37).sin()).collect();
    let ws: Vec<f32> = (0..wn).map(|i| 0.04 * (i % 23) as f32 - 0.45).collect();
    let mut g = Graph::new("conv3d");
    let x = g.input("x", sh(&[c.n, c.cin, c.dims[0], c.dims[1], c.dims[2]]));
    let w = g.input("w", sh(&[c.cout, c.cin / c.groups, c.k[0], c.k[1], c.k[2]]));
    let y = g.conv3d(x, w, c.s, c.p, c.d, c.groups);
    g.set_outputs(vec![y]);
    let plan = rlx_webgl::build_plan(&g).expect("plan");
    let inputs: [(&str, &[f32]); 2] = [("x", &xs), ("w", &ws)];
    let want = rlx_webgl::run_cpu(&plan, &inputs)
        .expect("run_cpu")
        .remove(0);
    let gl = rlx_webgl::exec_gl::GlBackend::new().expect("webgl2");
    let got = gl.run(&plan, &inputs).expect("run gl").remove(0);
    let label = format!(
        "n{} c{}->{} {:?} k{:?} s{:?} p{:?} d{:?} g{}",
        c.n, c.cin, c.cout, c.dims, c.k, c.s, c.p, c.d, c.groups
    );
    assert_eq!(got.len(), want.len(), "{label}");
    assert!(!want.is_empty(), "{label}: empty output");
    for (i, (a, e)) in got.iter().zip(&want).enumerate() {
        assert!(
            (a - e).abs() <= 1e-5 * (1.0 + e.abs()),
            "{label} [{i}]: webgl {a} cpu {e}"
        );
    }
}

fn conv(
    cin: usize,
    cout: usize,
    dims: [usize; 3],
    k: usize,
    s: [usize; 3],
    p: usize,
    d: usize,
) -> Conv {
    Conv {
        n: 1,
        cin,
        cout,
        dims,
        k: [k; 3],
        s,
        p: [p; 3],
        d: [d; 3],
        groups: 1,
    }
}

#[wasm_bindgen_test]
fn strided() {
    run(&conv(3, 4, [7, 8, 9], 3, [2; 3], 1, 1));
    run(&conv(2, 6, [9, 7, 10], 3, [3; 3], 1, 1));
    run(&conv(3, 5, [8, 8, 8], 2, [2; 3], 0, 1)); // a 2× downsampling conv
    run(&conv(4, 8, [6, 7, 11], 3, [2; 3], 2, 2)); // strided and dilated
    run(&conv(2, 4, [5, 5, 5], 3, [4; 3], 3, 1)); // stride wider than the kernel
}

#[wasm_bindgen_test]
fn anisotropic_stride_kernel_and_padding() {
    run(&conv(3, 4, [6, 9, 12], 3, [1, 2, 3], 1, 1));
    run(&conv(3, 4, [6, 9, 12], 3, [3, 1, 2], 2, 1));
    run(&Conv {
        n: 1,
        cin: 2,
        cout: 5,
        dims: [5, 8, 11],
        k: [1, 3, 5],
        s: [1, 2, 3],
        p: [0, 1, 2],
        d: [1, 1, 2],
        groups: 1,
    });
}

#[wasm_bindgen_test]
fn channel_quads_and_batches() {
    // 1, 2, 3, 4 targets; two passes of 12, of 12 + a short one, of 16
    for cout in [4, 5, 6, 7, 9, 13, 17, 20, 30] {
        run(&conv(3, cout, [4, 5, 6], 3, [1; 3], 1, 1));
    }
    run(&Conv {
        n: 2,
        ..conv(3, 6, [5, 6, 7], 3, [2; 3], 1, 1)
    });
    run(&Conv {
        n: 3,
        ..conv(2, 5, [4, 4, 5], 3, [1; 3], 1, 1)
    });
}

#[wasm_bindgen_test]
fn dilation_and_padding_wider_than_the_volume() {
    // MeshNet's dilation 16 on a small window: taps entirely in padding
    run(&conv(3, 5, [6, 5, 7], 3, [1; 3], 16, 16));
    run(&conv(2, 4, [7, 6, 9], 3, [2; 3], 8, 4));
}

#[wasm_bindgen_test]
fn one_channel_kernel() {
    run(&conv(3, 3, [5, 6, 7], 3, [2; 3], 1, 1)); // cout < 4
    run(&Conv {
        groups: 2,
        ..conv(4, 6, [5, 6, 7], 3, [2; 3], 1, 1)
    });
}

/// Activations folded into the step that writes their input: a biased conv's
/// unpack (6 channels, the multi-channel kernel) and the one-channel
/// kernel's store (3 channels) apply an ELU, an instance norm's apply a GELU.
/// Tall enough that the reshapes around the norm are aliases.
#[wasm_bindgen_test]
fn activations_folded_into_conv_and_group_norm() {
    use rlx_ir::Op;
    use rlx_ir::op::Activation;
    let sh = |d: &[usize]| Shape::new(d, DType::F32);
    let (ci, d) = (2usize, [64usize, 48, 6]);
    let sp = d.iter().product::<usize>();
    for co in [6usize, 3] {
        let xs: Vec<f32> = (0..ci * sp).map(|i| (i as f32 * 0.37).sin()).collect();
        let ws: Vec<f32> = (0..co * ci * 27)
            .map(|i| 0.04 * (i % 23) as f32 - 0.45)
            .collect();
        let bs: Vec<f32> = (0..co).map(|i| i as f32 * 0.1 - 0.2).collect();
        let gs: Vec<f32> = (0..co).map(|i| 1.0 + i as f32 * 0.2).collect();
        let bts: Vec<f32> = (0..co).map(|i| i as f32 * 0.05).collect();
        let vol = sh(&[1, co, d[0], d[1], d[2]]);
        let mut g = Graph::new("meshnet_layer");
        let x = g.input("x", sh(&[1, ci, d[0], d[1], d[2]]));
        let w = g.input("w", sh(&[co, ci, 3, 3, 3]));
        let b = g.input("b", sh(&[co]));
        let y = g.add_node(
            Op::FusedConvBiasAct {
                kernel_size: vec![3; 3],
                stride: vec![1; 3],
                padding: vec![2; 3],
                dilation: vec![2; 3],
                groups: 1,
                activation: None,
                has_residual: false,
            },
            vec![x, w, b],
            vol.clone(),
        );
        let a = g.add_node(Op::Activation(Activation::Elu), vec![y], vol.clone());
        let f = g.reshape(a, vec![1, co as i64, sp as i64, 1], sh(&[1, co, sp, 1]));
        let gamma = g.input("g", sh(&[co]));
        let beta = g.input("bt", sh(&[co]));
        let n = g.group_norm(f, gamma, beta, co, 1e-5);
        let r = g.reshape(
            n,
            vec![1, co as i64, d[0] as i64, d[1] as i64, d[2] as i64],
            vol.clone(),
        );
        let out = g.add_node(Op::Activation(Activation::Gelu), vec![r], vol.clone());
        g.set_outputs(vec![out]);
        let plan = rlx_webgl::build_plan(&g).expect("plan");
        assert!(
            !plan
                .steps
                .iter()
                .any(|s| matches!(s, rlx_webgl::Step::Unary { .. })),
            "c{co}: an activation was not folded"
        );
        let inputs: [(&str, &[f32]); 5] =
            [("x", &xs), ("w", &ws), ("b", &bs), ("g", &gs), ("bt", &bts)];
        let want = rlx_webgl::run_cpu(&plan, &inputs)
            .expect("run_cpu")
            .remove(0);
        let gl = rlx_webgl::exec_gl::GlBackend::new().expect("webgl2");
        let got = gl.run(&plan, &inputs).expect("run gl").remove(0);
        assert_eq!(got.len(), want.len());
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, e)| (a - e).abs() / (1.0 + e.abs()))
            .fold(0f32, f32::max);
        // the norm's statistics are summed in a different order on the GPU
        assert!(worst <= 1e-4, "c{co}: worst relative error {worst}");
    }
}
