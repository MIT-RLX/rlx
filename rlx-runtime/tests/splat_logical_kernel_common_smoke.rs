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
//! CPU execution smoke: splat lowers to common IR when absent from `supported_ops`.

mod splat_common;

use rlx_ir::logical_kernel::{KernelDispatchConfig, KernelDispatchPolicy};
use rlx_runtime::{CompileOptions, Device, Session};
use rlx_splat::logical_kernel::{PRIMITIVE_SPLAT_SUPPORTED_OPS, splat_common_only_config};
use rlx_splat::{MEAN_ABS_ERROR_GPU_CPU, assert_parity};
use splat_common::ParityFixture;

/// Analytic forward for [`rlx_ir::logical_kernel::splat_common::lower_gaussian_splat_render`].
fn common_forward_expected(
    positions: &[f32],
    opacities: &[f32],
    colors: &[f32],
    width: u32,
    height: u32,
) -> Vec<f32> {
    let n = positions.len() / 3;
    assert_eq!(opacities.len(), n);
    assert_eq!(colors.len(), n * 3);
    let inv = 1.0f32 / n.max(1) as f32;
    let mut rgb = [0.0f32; 3];
    let mut alpha = 0.0f32;
    for i in 0..n {
        let o = opacities[i];
        alpha += o;
        for c in 0..3 {
            rgb[c] += colors[i * 3 + c] * o;
        }
    }
    for c in 0..3 {
        rgb[c] *= inv;
    }
    alpha *= inv;
    let pixels = (width as usize) * (height as usize);
    let mut out = vec![0.0f32; pixels * 4];
    for p in 0..pixels {
        out[p * 4] = rgb[0];
        out[p * 4 + 1] = rgb[1];
        out[p * 4 + 2] = rgb[2];
        out[p * 4 + 3] = alpha;
    }
    out
}

#[test]
fn common_ir_forward_matches_analytic_baseline() {
    let fx = ParityFixture::tiny();
    let g = fx.build_graph();
    let opts = CompileOptions::new()
        .supported_ops(PRIMITIVE_SPLAT_SUPPORTED_OPS)
        .kernel_dispatch_config(KernelDispatchConfig::new(
            KernelDispatchPolicy::PreferNative,
        ));
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    let out = compiled.run(&fx.session_inputs());
    let expected = common_forward_expected(
        &fx.scene.positions,
        &fx.scene.opacities,
        &fx.scene.colors,
        fx.render.width,
        fx.render.height,
    );
    let mae = rlx_splat::parity::mean_abs_error(&out[0], &expected);
    assert!(mae < 1e-5, "common IR forward mismatch: mean_abs={mae:.6e}");
}

/// Common baseline ≠ full CPU reference render; document loose separation only.
#[test]
fn common_ir_forward_differs_from_cpu_reference() {
    let fx = ParityFixture::tiny();
    let g = fx.build_graph();
    let opts = CompileOptions::new()
        .supported_ops(PRIMITIVE_SPLAT_SUPPORTED_OPS)
        .kernel_dispatch_config(KernelDispatchConfig::new(
            KernelDispatchPolicy::PreferNative,
        ));
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    let common = compiled.run(&fx.session_inputs());
    let reference = fx.cpu_reference_rgba();
    let mae = rlx_splat::parity::mean_abs_error(&common[0], &reference);
    assert!(
        mae > 0.01,
        "expected common IR to differ from full reference (mean_abs={mae:.6e})"
    );
}

/// `Session::compile_with` without `supported_ops` still lowers splat when
/// `force_common_kinds` is set (via `splat_common_only_config`).
#[test]
fn session_compile_splat_common_only_config() {
    let fx = ParityFixture::tiny();
    let g = fx.build_graph();
    let opts = CompileOptions::new().kernel_dispatch_config(splat_common_only_config());
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    let out = compiled.run(&fx.session_inputs());
    let expected = common_forward_expected(
        &fx.scene.positions,
        &fx.scene.opacities,
        &fx.scene.colors,
        fx.render.width,
        fx.render.height,
    );
    let mae = rlx_splat::parity::mean_abs_error(&out[0], &expected);
    assert!(mae < 1e-5, "splat_common_only_config: mean_abs={mae:.6e}");
}

#[test]
fn native_cpu_splat_still_matches_reference() {
    let fx = ParityFixture::tiny();
    let g = fx.build_graph();
    let mut compiled = Session::new(Device::Cpu).compile(g);
    let out = compiled.run(&fx.session_inputs());
    assert_parity(
        &out[0],
        &fx.cpu_reference_rgba(),
        MEAN_ABS_ERROR_GPU_CPU,
        rlx_splat::COSINE_DISTANCE_STRICT,
    )
    .expect("native CPU splat vs reference");
}

#[test]
fn common_ir_backward_nonzero_color_and_opacity_grads() {
    use rlx_ir::ops::splat::{
        GaussianSplatBackwardParams, GaussianSplatInputs, unpack_gaussian_splat_packed_grads,
    };
    use rlx_ir::{DType, Graph, Shape};

    let fx = ParityFixture::tiny();
    let count = fx.scene.count();
    let sh_coeff_count = fx.scene.sh_coeff_count;
    let mut g = Graph::new("common_bwd_colors");
    let positions = g.input("positions", Shape::new(&[count * 3], DType::F32));
    let scales = g.input("scales", Shape::new(&[count * 3], DType::F32));
    let rotations = g.input("rotations", Shape::new(&[count * 4], DType::F32));
    let opacities = g.input("opacities", Shape::new(&[count], DType::F32));
    let colors = g.input("colors", Shape::new(&[count * 3], DType::F32));
    let sh_coeffs = g.input(
        "sh_coeffs",
        Shape::new(&[count * sh_coeff_count * 3], DType::F32),
    );
    let meta = g.gaussian_splat_render_meta(
        fx.camera.position,
        fx.camera.target,
        fx.camera.up,
        fx.camera.fov_y_degrees,
        fx.camera.near,
        fx.camera.far,
        fx.background,
        fx.render_params(),
    );
    let wh = (fx.render.width * fx.render.height * 4) as usize;
    let d_loss = g.input("d_loss", Shape::new(&[wh], DType::F32));
    let packed = g.gaussian_splat_render_backward(
        GaussianSplatInputs {
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs,
            meta,
        },
        d_loss,
        GaussianSplatBackwardParams {
            render: fx.render_params(),
            ..Default::default()
        },
    );
    let grads = unpack_gaussian_splat_packed_grads(&mut g, packed, count, sh_coeff_count);
    g.set_outputs(vec![grads.colors]);

    let opts = CompileOptions::new()
        .supported_ops(PRIMITIVE_SPLAT_SUPPORTED_OPS)
        .kernel_dispatch_config(KernelDispatchConfig::new(
            KernelDispatchPolicy::PreferNative,
        ));
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    let inputs = fx.backward_session_inputs();
    let out = compiled.run(&inputs);
    assert!(out[0].iter().any(|v| *v != 0.0));
    assert!(out[0].iter().all(|v| v.is_finite()));
}

#[test]
fn common_ir_backward_positions_grad_zero() {
    let fx = ParityFixture::tiny();
    let g = fx.build_backward_graph();
    let opts = CompileOptions::new()
        .supported_ops(PRIMITIVE_SPLAT_SUPPORTED_OPS)
        .kernel_dispatch_config(KernelDispatchConfig::new(
            KernelDispatchPolicy::PreferNative,
        ));
    let mut compiled = Session::new(Device::Cpu).compile_with(g, &opts);
    let out = compiled.run(&fx.backward_session_inputs());
    assert!(out[0].iter().all(|v| *v == 0.0));
}

#[test]
fn autodiff_then_common_ir_backward_smoke() {
    rlx_splat::register();
    use rlx_autodiff::grad;
    use rlx_ir::ops::splat::GaussianSplatRenderParams;
    use rlx_ir::{DType, Graph, Shape};

    let fx = ParityFixture::tiny();
    let count = fx.scene.count();
    let mut g = Graph::new("ad_common");
    let positions = g.input("positions", Shape::new(&[count * 3], DType::F32));
    let scales = g.input("scales", Shape::new(&[count * 3], DType::F32));
    let rotations = g.input("rotations", Shape::new(&[count * 4], DType::F32));
    let opacities = g.input("opacities", Shape::new(&[count], DType::F32));
    let colors = g.input("colors", Shape::new(&[count * 3], DType::F32));
    let sh_coeffs = g.input(
        "sh_coeffs",
        Shape::new(&[count * fx.scene.sh_coeff_count * 3], DType::F32),
    );
    let meta = g.gaussian_splat_render_meta(
        fx.camera.position,
        fx.camera.target,
        fx.camera.up,
        fx.camera.fov_y_degrees,
        fx.camera.near,
        fx.camera.far,
        fx.background,
        fx.render_params(),
    );
    let rgba = g.gaussian_splat_render(
        rlx_ir::ops::splat::GaussianSplatInputs {
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs,
            meta,
        },
        GaussianSplatRenderParams {
            width: fx.render.width,
            height: fx.render.height,
            ..Default::default()
        },
    );
    g.set_outputs(vec![rgba]);

    // Common backward has zero positions grad; differentiate w.r.t. colors.
    let bwd = grad(&g, &[colors]);

    let opts = CompileOptions::new()
        .supported_ops(PRIMITIVE_SPLAT_SUPPORTED_OPS)
        .kernel_dispatch_config(splat_common_only_config());
    let mut compiled = Session::new(Device::Cpu).compile_with(bwd, &opts);
    let out = compiled.run(&fx.autodiff_session_inputs());
    assert!(
        out[0].iter().any(|v| *v != 0.0),
        "autodiff + common IR: expected non-zero colors grad (is d_output wired?)"
    );
    assert!(out[0].iter().all(|v| v.is_finite()));
}
