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

//! 3D Gaussian splatting graph builders.

use crate::infer::GraphExt;
use crate::op::Op;
use crate::{DType, Graph, NodeId, Shape};

/// Packed scene + camera tensors for [`Op::GaussianSplatRender`].
#[derive(Clone, Debug)]
pub struct GaussianSplatInputs {
    pub positions: NodeId,
    pub scales: NodeId,
    pub rotations: NodeId,
    pub opacities: NodeId,
    pub colors: NodeId,
    pub sh_coeffs: NodeId,
    pub meta: NodeId,
}

/// Render-parameter bundle (embedded in the op, not separate tensors).
#[derive(Clone, Copy, Debug)]
pub struct GaussianSplatRenderParams {
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub radius_scale: f32,
    pub alpha_cutoff: f32,
    pub max_splat_steps: u32,
    pub transmittance_threshold: f32,
    pub max_list_entries: u32,
}

/// Training backward parameters for [`Op::GaussianSplatRenderBackward`].
#[derive(Clone, Copy, Debug)]
pub struct GaussianSplatBackwardParams {
    pub render: GaussianSplatRenderParams,
    pub loss_grad_clip: f32,
    pub sh_band: u32,
    pub max_anisotropy: f32,
}

impl Default for GaussianSplatBackwardParams {
    fn default() -> Self {
        Self {
            render: GaussianSplatRenderParams::default(),
            loss_grad_clip: 1.0,
            sh_band: 0,
            max_anisotropy: 10.0,
        }
    }
}

/// Trailing raster params in a packed prepare buffer (`width`, `height`, …).
pub const GAUSSIAN_SPLAT_PREP_RASTER_PARAMS_FLOATS: usize = 11;

/// Tile count for a framebuffer (matches `rlx_splat::prep_layout::tile_count`).
pub fn gaussian_splat_tile_count(width: u32, height: u32, tile_size: u32) -> u32 {
    let tw = (width + tile_size - 1) / tile_size;
    let th = (height + tile_size - 1) / tile_size;
    tw * th
}

/// Packed prepare-buffer length for `N` splats (must match `rlx_splat::prep_layout::pack_prepared`).
pub fn gaussian_splat_prep_packed_len(
    count: usize,
    max_list_entries: u32,
    width: u32,
    height: u32,
    tile_size: u32,
) -> usize {
    let n = count.max(1);
    let max_list = max_list_entries as usize;
    let tiles = gaussian_splat_tile_count(width, height, tile_size) as usize;
    let pixels = (width as usize).saturating_mul(height as usize).max(1);
    n * 4
        + n
        + n * 3
        + n * 3
        + n * 4
        + max_list
        + tiles * 2
        + pixels * 3
        + GAUSSIAN_SPLAT_PREP_RASTER_PARAMS_FLOATS
}

/// Packed scene gradient layout lengths for `N` splats and `sh_coeff_count` SH bands.
pub fn gaussian_splat_packed_grad_len(count: usize, sh_coeff_count: usize) -> usize {
    count * (3 + 3 + 4 + 1 + 3) + count * sh_coeff_count.max(1) * 3
}

/// Unpack [`Op::GaussianSplatRenderBackward`] output into per-parameter gradients.
pub fn unpack_gaussian_splat_packed_grads(
    g: &mut Graph,
    packed: NodeId,
    count: usize,
    sh_coeff_count: usize,
) -> GaussianSplatInputs {
    let mut off = 0usize;
    let mut take = |len: usize| -> NodeId {
        let id = g.narrow_(packed, 0, off, len);
        off += len;
        id
    };
    let positions = take(count * 3);
    let scales = take(count * 3);
    let rotations = take(count * 4);
    let opacities = take(count);
    let colors = take(count * 3);
    let sh_coeffs = take(count * sh_coeff_count.max(1) * 3);
    let _ = off;
    GaussianSplatInputs {
        positions,
        scales,
        rotations,
        opacities,
        colors,
        sh_coeffs,
        meta: packed,
    }
}

impl Default for GaussianSplatRenderParams {
    fn default() -> Self {
        Self {
            width: 64,
            height: 64,
            tile_size: 16,
            radius_scale: 1.6,
            alpha_cutoff: 1.0 / 255.0,
            max_splat_steps: 32,
            transmittance_threshold: 0.01,
            max_list_entries: 18 * 32,
        }
    }
}

impl Graph {
    /// First-class CPU reference Gaussian splat forward render.
    ///
    /// See [`Op::GaussianSplatRender`] for the seven-input contract and
    /// [`GaussianSplatRenderParams`] for framebuffer settings.
    pub fn gaussian_splat_render(
        &mut self,
        inputs: GaussianSplatInputs,
        params: GaussianSplatRenderParams,
    ) -> NodeId {
        let out_elems = (params.width as usize) * (params.height as usize) * 4;
        let dtype = self.shape(inputs.positions).dtype();
        let out_shape = Shape::new(&[out_elems], dtype);
        self.push(
            Op::GaussianSplatRender {
                width: params.width,
                height: params.height,
                tile_size: params.tile_size,
                radius_scale: params.radius_scale,
                alpha_cutoff: params.alpha_cutoff,
                max_splat_steps: params.max_splat_steps,
                transmittance_threshold: params.transmittance_threshold,
                max_list_entries: params.max_list_entries,
            },
            vec![
                inputs.positions,
                inputs.scales,
                inputs.rotations,
                inputs.opacities,
                inputs.colors,
                inputs.sh_coeffs,
                inputs.meta,
            ],
            out_shape,
            None,
        )
    }

    /// Build the 23-float `meta` vector expected by [`Op::GaussianSplatRender`].
    pub fn gaussian_splat_render_meta(
        &mut self,
        camera_position: [f32; 3],
        camera_target: [f32; 3],
        camera_up: [f32; 3],
        fov_y_degrees: f32,
        near: f32,
        far: f32,
        background: [f32; 3],
        params: GaussianSplatRenderParams,
    ) -> NodeId {
        let values = vec![
            camera_position[0],
            camera_position[1],
            camera_position[2],
            camera_target[0],
            camera_target[1],
            camera_target[2],
            camera_up[0],
            camera_up[1],
            camera_up[2],
            fov_y_degrees,
            near,
            far,
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
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(&[23], DType::F32),
        )
    }

    /// Strict IR stage 1: project + bin + sort + rays → packed prepare buffer.
    pub fn gaussian_splat_prepare(
        &mut self,
        inputs: GaussianSplatInputs,
        params: GaussianSplatRenderParams,
    ) -> NodeId {
        let count = self.shape(inputs.positions).num_elements().unwrap_or(0) / 3;
        let packed_len = gaussian_splat_prep_packed_len(
            count,
            params.max_list_entries,
            params.width,
            params.height,
            params.tile_size,
        );
        let dtype = self.shape(inputs.positions).dtype();
        self.push(
            Op::GaussianSplatPrepare {
                width: params.width,
                height: params.height,
                tile_size: params.tile_size,
                radius_scale: params.radius_scale,
                alpha_cutoff: params.alpha_cutoff,
                max_splat_steps: params.max_splat_steps,
                transmittance_threshold: params.transmittance_threshold,
                max_list_entries: params.max_list_entries,
            },
            vec![
                inputs.positions,
                inputs.scales,
                inputs.rotations,
                inputs.opacities,
                inputs.colors,
                inputs.sh_coeffs,
                inputs.meta,
            ],
            Shape::new(&[packed_len], dtype),
            None,
        )
    }

    /// Strict IR stage 2: rasterize from prepare buffer + meta.
    pub fn gaussian_splat_rasterize(
        &mut self,
        prep: NodeId,
        meta: NodeId,
        params: GaussianSplatRenderParams,
    ) -> NodeId {
        let out_elems = (params.width as usize) * (params.height as usize) * 4;
        let dtype = self.shape(prep).dtype();
        self.push(
            Op::GaussianSplatRasterize {
                width: params.width,
                height: params.height,
                tile_size: params.tile_size,
                alpha_cutoff: params.alpha_cutoff,
                max_splat_steps: params.max_splat_steps,
                transmittance_threshold: params.transmittance_threshold,
                max_list_entries: params.max_list_entries,
            },
            vec![prep, meta],
            Shape::new(&[out_elems], dtype),
            None,
        )
    }

    /// Decomposed strict-IR forward: prepare → rasterize.
    pub fn gaussian_splat_render_decomposed(
        &mut self,
        inputs: GaussianSplatInputs,
        params: GaussianSplatRenderParams,
    ) -> NodeId {
        let meta = inputs.meta;
        let prep = self.gaussian_splat_prepare(inputs, params);
        self.gaussian_splat_rasterize(prep, meta, params)
    }

    /// Backward pass for [`Op::GaussianSplatRender`] (packed scene gradients).
    pub fn gaussian_splat_render_backward(
        &mut self,
        inputs: GaussianSplatInputs,
        d_loss_rgba: NodeId,
        params: GaussianSplatBackwardParams,
    ) -> NodeId {
        let count = self.shape(inputs.positions).num_elements().unwrap_or(0) / 3;
        let sh_len = self.shape(inputs.sh_coeffs).num_elements().unwrap_or(0);
        let sh_coeff_count = if count == 0 {
            1
        } else {
            (sh_len / (count * 3)).max(1)
        };
        let packed_len = gaussian_splat_packed_grad_len(count, sh_coeff_count);
        let dtype = self.shape(inputs.positions).dtype();
        let r = params.render;
        self.push(
            Op::GaussianSplatRenderBackward {
                width: r.width,
                height: r.height,
                tile_size: r.tile_size,
                radius_scale: r.radius_scale,
                alpha_cutoff: r.alpha_cutoff,
                max_splat_steps: r.max_splat_steps,
                transmittance_threshold: r.transmittance_threshold,
                max_list_entries: r.max_list_entries,
                loss_grad_clip: params.loss_grad_clip,
                sh_band: params.sh_band,
                max_anisotropy: params.max_anisotropy,
            },
            vec![
                inputs.positions,
                inputs.scales,
                inputs.rotations,
                inputs.opacities,
                inputs.colors,
                inputs.sh_coeffs,
                inputs.meta,
                d_loss_rgba,
            ],
            Shape::new(&[packed_len], dtype),
            None,
        )
    }
}
