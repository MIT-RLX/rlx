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
//! RLX custom-op registration for Gaussian splat CPU reference rendering.

use std::sync::Arc;

use rlx_ir::{Node, NodeId, OpExtension, Shape, VjpContext, register_op};

#[cfg(feature = "cpu")]
use rlx_cpu::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef, register_cpu_kernel};

use crate::core::Camera;
use crate::reference::{RenderParams, render_reference};

/// Canonical op name for new RLX graphs.
pub const RENDER_REFERENCE: &str = "rlx_splat.render_reference";

/// Legacy custom-op alias (deprecated).
pub const RENDER_REFERENCE_LEGACY: &str = "rlx_splat.render_reference_v0";

struct RenderReferenceExt {
    name: &'static str,
}

impl RenderReferenceExt {
    const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

impl OpExtension for RenderReferenceExt {
    fn name(&self) -> &str {
        self.name
    }

    fn num_inputs(&self) -> usize {
        7
    }

    fn infer_shape(&self, inputs: &[&Shape], attrs: &[u8]) -> Shape {
        let _ = inputs;
        let width = u32::from_le_bytes(attrs[0..4].try_into().unwrap()) as usize;
        let height = u32::from_le_bytes(attrs[4..8].try_into().unwrap()) as usize;
        Shape::new(&[height * width * 4], rlx_ir::DType::F32)
    }

    fn vjp(&self, _node: &Node, _ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        Vec::new()
    }
}

#[cfg(feature = "cpu")]
struct RenderReferenceCpu {
    name: &'static str,
}

#[cfg(feature = "cpu")]
impl RenderReferenceCpu {
    const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

#[cfg(feature = "cpu")]
impl CpuKernel for RenderReferenceCpu {
    fn name(&self) -> &str {
        self.name
    }

    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let positions = inputs[0].expect_f32("positions")?;
        let scales = inputs[1].expect_f32("scales")?;
        let rotations = inputs[2].expect_f32("rotations")?;
        let opacities = inputs[3].expect_f32("opacities")?;
        let colors = inputs[4].expect_f32("colors")?;
        let sh_coeffs = inputs[5].expect_f32("sh_coeffs")?;
        let meta = inputs[6].expect_f32("meta")?;
        let count = positions.len() / 3;
        let sh_coeff_count = if count == 0 {
            1
        } else {
            sh_coeffs.len() / (count * 3).max(1)
        };
        let scene = crate::core::GaussianScene::new(
            positions.to_vec(),
            scales.to_vec(),
            rotations.to_vec(),
            opacities.to_vec(),
            colors.to_vec(),
            sh_coeffs.to_vec(),
            sh_coeff_count.max(1),
        );
        let camera = Camera::look_at(
            [meta[0], meta[1], meta[2]],
            [meta[3], meta[4], meta[5]],
            [meta[6], meta[7], meta[8]],
            meta[9],
            meta[10],
            meta[11],
        );
        let background = [meta[12], meta[13], meta[14]];
        let params = RenderParams {
            width: meta[15] as u32,
            height: meta[16] as u32,
            tile_size: meta[17] as u32,
            radius_scale: meta[18],
            alpha_cutoff: meta[19],
            max_splat_steps: meta[20] as u32,
            transmittance_threshold: meta[21],
            max_list_entries: meta[22] as u32,
        };
        let image = render_reference(&scene, &camera, background, &params);
        let out = output.expect_f32_mut("rgba")?;
        if out.len() != image.len() {
            return Err(format!(
                "render_reference: output len {} != image len {}",
                out.len(),
                image.len()
            ));
        }
        out.copy_from_slice(&image);
        let _ = attrs;
        Ok(())
    }
}

pub fn register() {
    register_op(Arc::new(RenderReferenceExt::new(RENDER_REFERENCE)));
    register_op(Arc::new(RenderReferenceExt::new(RENDER_REFERENCE_LEGACY)));
    #[cfg(feature = "cpu")]
    {
        register_cpu_kernel(Arc::new(RenderReferenceCpu::new(RENDER_REFERENCE)));
        register_cpu_kernel(Arc::new(RenderReferenceCpu::new(RENDER_REFERENCE_LEGACY)));
    }
}

pub fn encode_render_attrs(width: u32, height: u32) -> Vec<u8> {
    let mut attrs = Vec::with_capacity(8);
    attrs.extend_from_slice(&width.to_le_bytes());
    attrs.extend_from_slice(&height.to_le_bytes());
    attrs
}

pub fn build_render_meta(camera: &Camera, background: [f32; 3], params: &RenderParams) -> Vec<f32> {
    vec![
        camera.position[0],
        camera.position[1],
        camera.position[2],
        camera.target[0],
        camera.target[1],
        camera.target[2],
        camera.up[0],
        camera.up[1],
        camera.up[2],
        camera.fov_y_degrees,
        camera.near,
        camera.far,
        background[0],
        background[1],
        background[2],
        params.width as f32,
        params.height as f32,
        params.tile_size as f32,
        params.radius_scale,
        params.alpha_cutoff,
        params.max_splat_steps as f32,
        params.transmittance_threshold,
        params.max_list_entries as f32,
    ]
}
