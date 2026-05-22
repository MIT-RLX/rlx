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
//! Shared parity scene + graph builder for cross-backend integration tests.

use rlx_ir::ops::splat::{
    GaussianSplatBackwardParams, GaussianSplatInputs, GaussianSplatRenderParams,
    unpack_gaussian_splat_packed_grads,
};
use rlx_ir::{DType, Graph, Shape};
use rlx_splat::core::{Camera, GaussianScene};
use rlx_splat::reference::RenderParams;
use rlx_splat::{
    make_parity_scene, parity_camera, parity_tiny_render_params, PARITY_BACKGROUND,
};

pub struct ParityFixture {
    pub scene: GaussianScene,
    pub camera: Camera,
    pub render: RenderParams,
    pub background: [f32; 3],
}

impl ParityFixture {
    pub fn tiny() -> Self {
        let scene = make_parity_scene();
        Self {
            scene,
            camera: parity_camera(),
            render: parity_tiny_render_params(),
            background: PARITY_BACKGROUND,
        }
    }

    pub fn build_graph(&self) -> Graph {
        let mut g = Graph::new("gaussian_splat_backend_test");
        let count = self.scene.count();
        let sh_coeff_count = self.scene.sh_coeff_count;
        let positions = g.input("positions", Shape::new(&[count * 3], DType::F32));
        let scales = g.input("scales", Shape::new(&[count * 3], DType::F32));
        let rotations = g.input("rotations", Shape::new(&[count * 4], DType::F32));
        let opacities = g.input("opacities", Shape::new(&[count], DType::F32));
        let colors = g.input("colors", Shape::new(&[count * 3], DType::F32));
        let sh_coeffs = g.input(
            "sh_coeffs",
            Shape::new(&[count * sh_coeff_count * 3], DType::F32),
        );
        let rgba = rlx_splat::gaussian_splat_render_scene(
            &mut g,
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs,
            &self.camera,
            self.background,
            &self.render,
        );
        g.set_outputs(vec![rgba]);
        g
    }

    pub fn session_inputs(&self) -> [(&str, &[f32]); 6] {
        [
            ("positions", &self.scene.positions),
            ("scales", &self.scene.scales),
            ("rotations", &self.scene.rotations),
            ("opacities", &self.scene.opacities),
            ("colors", &self.scene.colors),
            ("sh_coeffs", &self.scene.sh_coeffs),
        ]
    }

    pub fn cpu_reference_rgba(&self) -> Vec<f32> {
        rlx_splat::reference::render_reference(
            &self.scene,
            &self.camera,
            self.background,
            &self.render,
        )
    }

    pub fn render_params(&self) -> GaussianSplatRenderParams {
        GaussianSplatRenderParams {
            width: self.render.width,
            height: self.render.height,
            tile_size: self.render.tile_size,
            radius_scale: self.render.radius_scale,
            alpha_cutoff: self.render.alpha_cutoff,
            max_splat_steps: self.render.max_splat_steps,
            transmittance_threshold: self.render.transmittance_threshold,
            max_list_entries: self.render.max_list_entries,
        }
    }

    /// Graph with [`Op::GaussianSplatRenderBackward`]; output is positions grad.
    pub fn build_backward_graph(&self) -> Graph {
        let mut g = Graph::new("gaussian_splat_backward_test");
        let count = self.scene.count();
        let sh_coeff_count = self.scene.sh_coeff_count;
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
            self.camera.position,
            self.camera.target,
            self.camera.up,
            self.camera.fov_y_degrees,
            self.camera.near,
            self.camera.far,
            self.background,
            self.render_params(),
        );
        let inputs = GaussianSplatInputs {
            positions,
            scales,
            rotations,
            opacities,
            colors,
            sh_coeffs,
            meta,
        };
        let wh = (self.render.width * self.render.height * 4) as usize;
        let d_loss = g.input("d_loss", Shape::new(&[wh], DType::F32));
        let packed = g.gaussian_splat_render_backward(
            inputs,
            d_loss,
            GaussianSplatBackwardParams {
                render: self.render_params(),
                ..Default::default()
            },
        );
        let grads = unpack_gaussian_splat_packed_grads(&mut g, packed, count, sh_coeff_count);
        g.set_outputs(vec![grads.positions]);
        g
    }

    /// Inputs for [`rlx_autodiff::grad`] graphs (includes `d_output` seed).
    pub fn autodiff_session_inputs(&self) -> [(&str, &[f32]); 7] {
        let wh = (self.render.width * self.render.height * 4) as usize;
        let d_output: &'static [f32] = Box::leak(vec![1.0f32; wh].into_boxed_slice());
        [
            ("positions", &self.scene.positions),
            ("scales", &self.scene.scales),
            ("rotations", &self.scene.rotations),
            ("opacities", &self.scene.opacities),
            ("colors", &self.scene.colors),
            ("sh_coeffs", &self.scene.sh_coeffs),
            ("d_output", d_output),
        ]
    }

    pub fn backward_session_inputs(&self) -> [(&str, &[f32]); 7] {
        let d_loss = vec![1.0f32; (self.render.width * self.render.height * 4) as usize];
        // Leak is test-only; inputs must live for `run`.
        let d_loss: &'static [f32] = Box::leak(d_loss.into_boxed_slice());
        [
            ("positions", &self.scene.positions),
            ("scales", &self.scene.scales),
            ("rotations", &self.scene.rotations),
            ("opacities", &self.scene.opacities),
            ("colors", &self.scene.colors),
            ("sh_coeffs", &self.scene.sh_coeffs),
            ("d_loss", d_loss),
        ]
    }

    /// CPU reference positions grad (same kernel as [`Op::GaussianSplatRenderBackward`]).
    pub fn cpu_reference_positions_grad(&self) -> Vec<f32> {
        let meta = [
            self.camera.position[0],
            self.camera.position[1],
            self.camera.position[2],
            self.camera.target[0],
            self.camera.target[1],
            self.camera.target[2],
            self.camera.up[0],
            self.camera.up[1],
            self.camera.up[2],
            self.camera.fov_y_degrees,
            self.camera.near,
            self.camera.far,
            self.background[0],
            self.background[1],
            self.background[2],
        ];
        let d_loss = vec![1.0f32; (self.render.width * self.render.height * 4) as usize];
        let packed = rlx_cpu::splat::backward_host_slices(
            &self.scene.positions,
            &self.scene.scales,
            &self.scene.rotations,
            &self.scene.opacities,
            &self.scene.colors,
            &self.scene.sh_coeffs,
            &meta,
            &d_loss,
            self.render.width,
            self.render.height,
            self.render.tile_size,
            self.render.radius_scale,
            self.render.alpha_cutoff,
            self.render.max_splat_steps,
            self.render.transmittance_threshold,
            self.render.max_list_entries,
            1.0,
            0,
            10.0,
        );
        let n = self.scene.count() * 3;
        packed[..n].to_vec()
    }

    /// Build a minimal graph using only [`GaussianSplatInputs`] (no scene helper).
    #[allow(dead_code)]
    pub fn build_graph_inline(&self) -> Graph {
        let mut g = Graph::new("gaussian_splat_inline");
        let count = self.scene.count();
        let sh_coeff_count = self.scene.sh_coeff_count;
        let shape_n = Shape::new(&[count * 3], DType::F32);
        let positions = g.input("positions", shape_n.clone());
        let scales = g.input("scales", shape_n.clone());
        let rotations = g.input("rotations", Shape::new(&[count * 4], DType::F32));
        let opacities = g.input("opacities", Shape::new(&[count], DType::F32));
        let colors = g.input("colors", shape_n.clone());
        let sh_coeffs = g.input(
            "sh_coeffs",
            Shape::new(&[count * sh_coeff_count * 3], DType::F32),
        );
        let params = GaussianSplatRenderParams {
            width: self.render.width,
            height: self.render.height,
            tile_size: self.render.tile_size,
            radius_scale: self.render.radius_scale,
            alpha_cutoff: self.render.alpha_cutoff,
            max_splat_steps: self.render.max_splat_steps,
            transmittance_threshold: self.render.transmittance_threshold,
            max_list_entries: self.render.max_list_entries,
        };
        let meta = g.gaussian_splat_render_meta(
            self.camera.position,
            self.camera.target,
            self.camera.up,
            self.camera.fov_y_degrees,
            self.camera.near,
            self.camera.far,
            self.background,
            params,
        );
        let rgba = g.gaussian_splat_render(
            GaussianSplatInputs {
                positions,
                scales,
                rotations,
                opacities,
                colors,
                sh_coeffs,
                meta,
            },
            params,
        );
        g.set_outputs(vec![rgba]);
        g
    }
}
