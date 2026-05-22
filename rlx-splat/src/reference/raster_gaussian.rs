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
//! Cached raster Gaussian layout (`RasterGaussian` / 13 floats per splat).

use crate::core::{
    evaluate_sh0_sh1, quat_rotate, resolve_supported_sh_coeffs, Camera, GaussianScene,
    GAUSSIAN_SUPPORT_SIGMA_RADIUS, VEC_EPS,
};

use crate::core::RASTER_CACHE_PARAM_COUNT;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CachedRasterGaussian {
    pub ro_camera: [f32; 3],
    pub sigma_diag: [f32; 3],
    pub sigma_off_diag: [f32; 3],
    pub color_alpha: [f32; 4],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CachedRasterGrad {
    pub ro_camera: [f32; 3],
    pub sigma_diag: [f32; 3],
    pub sigma_off_diag: [f32; 3],
    pub color_alpha: [f32; 4],
}

impl CachedRasterGrad {
    pub fn contribution_norm_sq(&self) -> f32 {
        self.ro_camera[0] * self.ro_camera[0]
            + self.ro_camera[1] * self.ro_camera[1]
            + self.sigma_diag[0] * self.sigma_diag[0]
            + self.sigma_diag[1] * self.sigma_diag[1]
            + self.sigma_off_diag[0] * self.sigma_off_diag[0]
            + self.color_alpha[0] * self.color_alpha[0]
            + self.color_alpha[1] * self.color_alpha[1]
            + self.color_alpha[2] * self.color_alpha[2]
            + self.color_alpha[3] * self.color_alpha[3]
    }

    pub fn accumulate(&mut self, other: &Self) {
        for i in 0..3 {
            self.ro_camera[i] += other.ro_camera[i];
            self.sigma_diag[i] += other.sigma_diag[i];
            self.sigma_off_diag[i] += other.sigma_off_diag[i];
        }
        for i in 0..4 {
            self.color_alpha[i] += other.color_alpha[i];
        }
    }
}

pub fn raster_gaussian_to_flat(g: &CachedRasterGaussian) -> [f32; RASTER_CACHE_PARAM_COUNT] {
    [
        g.ro_camera[0],
        g.ro_camera[1],
        g.ro_camera[2],
        g.sigma_diag[0],
        g.sigma_diag[1],
        g.sigma_diag[2],
        g.sigma_off_diag[0],
        g.sigma_off_diag[1],
        g.sigma_off_diag[2],
        g.color_alpha[0],
        g.color_alpha[1],
        g.color_alpha[2],
        g.color_alpha[3],
    ]
}

pub fn flat_to_raster_gaussian(flat: &[f32]) -> CachedRasterGaussian {
    CachedRasterGaussian {
        ro_camera: [flat[0], flat[1], flat[2]],
        sigma_diag: [flat[3], flat[4], flat[5]],
        sigma_off_diag: [flat[6], flat[7], flat[8]],
        color_alpha: [flat[9], flat[10], flat[11], flat[12]],
    }
}

fn rotate_by_quat(v: [f32; 3], q: [f32; 4]) -> [f32; 3] {
    quat_rotate(v, q)
}

fn world_to_local_dir(dir: [f32; 3], scale: [f32; 3], rot: [f32; 4]) -> [f32; 3] {
    let rotated = rotate_by_quat(dir, rot);
    [
        rotated[0] / scale[0].max(VEC_EPS),
        rotated[1] / scale[1].max(VEC_EPS),
        rotated[2] / scale[2].max(VEC_EPS),
    ]
}

/// Forward build aligned with `build_raster_gaussian` in `project.wgsl`.
pub fn build_raster_gaussian(
    scene: &GaussianScene,
    splat: usize,
    camera: &Camera,
    radius_scale: f32,
    max_anisotropy: f32,
    sh_band: u32,
) -> CachedRasterGaussian {
    let pos = scene.position(splat);
    let log_scale = scene.scale(splat);
    let rot = scene.rotation(splat);
    let sigma = [log_scale[0].exp(), log_scale[1].exp(), log_scale[2].exp()];
    let max_aniso = max_anisotropy.max(1.0);
    let raw_support = [
        sigma[0] * radius_scale * GAUSSIAN_SUPPORT_SIGMA_RADIUS,
        sigma[1] * radius_scale * GAUSSIAN_SUPPORT_SIGMA_RADIUS,
        sigma[2] * radius_scale * GAUSSIAN_SUPPORT_SIGMA_RADIUS,
    ];
    let support_max = raw_support[0].max(raw_support[1]).max(raw_support[2]);
    let support_min = support_max / max_aniso;
    let support_scale = [
        raw_support[0].max(support_min),
        raw_support[1].max(support_min),
        raw_support[2].max(support_min),
    ];

    let basis = camera.basis();
    let cam_to_world_x = [basis[0][0], basis[1][0], basis[2][0]];
    let cam_to_world_y = [basis[0][1], basis[1][1], basis[2][1]];
    let cam_to_world_z = [basis[0][2], basis[1][2], basis[2][2]];

    let axis_x = world_to_local_dir(cam_to_world_x, support_scale, rot);
    let axis_y = world_to_local_dir(cam_to_world_y, support_scale, rot);
    let axis_z = world_to_local_dir(cam_to_world_z, support_scale, rot);

    let delta = [
        camera.position[0] - pos[0],
        camera.position[1] - pos[1],
        camera.position[2] - pos[2],
    ];
    let ro_camera = camera.world_to_camera(delta);

    let sh_coeffs = resolve_supported_sh_coeffs(
        &scene.sh_coeffs,
        &scene.colors,
        scene.count(),
        scene.sh_coeff_count,
    );
    let view_dirs = [
        camera.position[0] - pos[0],
        camera.position[1] - pos[1],
        camera.position[2] - pos[2],
    ];
    let colors = evaluate_sh0_sh1(&sh_coeffs, &view_dirs, 1);
    let sh0 = [colors[0], colors[1], colors[2]];
    let _ = sh_band;

    CachedRasterGaussian {
        ro_camera,
        sigma_diag: [
            axis_x[0] * axis_x[0] + axis_x[1] * axis_x[1] + axis_x[2] * axis_x[2],
            axis_y[0] * axis_y[0] + axis_y[1] * axis_y[1] + axis_y[2] * axis_y[2],
            axis_z[0] * axis_z[0] + axis_z[1] * axis_z[1] + axis_z[2] * axis_z[2],
        ],
        sigma_off_diag: [
            axis_x[0] * axis_y[0] + axis_x[1] * axis_y[1] + axis_x[2] * axis_y[2],
            axis_x[0] * axis_z[0] + axis_x[1] * axis_z[1] + axis_x[2] * axis_z[2],
            axis_y[0] * axis_z[0] + axis_y[1] * axis_z[1] + axis_y[2] * axis_z[2],
        ],
        color_alpha: [sh0[0], sh0[1], sh0[2], scene.opacities[splat]],
    }
}

/// Numeric backprop from cached raster grad into scene parameter partials for one splat.
pub fn backprop_build_raster_gaussian(
    scene: &GaussianScene,
    splat: usize,
    camera: &Camera,
    radius_scale: f32,
    max_anisotropy: f32,
    sh_band: u32,
    d_raster: &CachedRasterGrad,
) -> ([f32; 3], [f32; 3], [f32; 4], f32) {
    let eps = 1e-4f32;
    let mut d_pos = [0.0f32; 3];
    let mut d_scale = [0.0f32; 3];
    let mut d_rot = [0.0f32; 4];
    let mut d_raw_opacity = 0.0f32;

    let base = build_raster_gaussian(scene, splat, camera, radius_scale, max_anisotropy, sh_band);
    let loss = |g: &CachedRasterGaussian| -> f32 {
        d_raster.ro_camera[0] * (g.ro_camera[0] - base.ro_camera[0])
            + d_raster.ro_camera[1] * (g.ro_camera[1] - base.ro_camera[1])
            + d_raster.ro_camera[2] * (g.ro_camera[2] - base.ro_camera[2])
            + d_raster.sigma_diag[0] * (g.sigma_diag[0] - base.sigma_diag[0])
            + d_raster.sigma_diag[1] * (g.sigma_diag[1] - base.sigma_diag[1])
            + d_raster.sigma_diag[2] * (g.sigma_diag[2] - base.sigma_diag[2])
            + d_raster.sigma_off_diag[0] * (g.sigma_off_diag[0] - base.sigma_off_diag[0])
            + d_raster.sigma_off_diag[1] * (g.sigma_off_diag[1] - base.sigma_off_diag[1])
            + d_raster.sigma_off_diag[2] * (g.sigma_off_diag[2] - base.sigma_off_diag[2])
            + d_raster.color_alpha[0] * (g.color_alpha[0] - base.color_alpha[0])
            + d_raster.color_alpha[1] * (g.color_alpha[1] - base.color_alpha[1])
            + d_raster.color_alpha[2] * (g.color_alpha[2] - base.color_alpha[2])
            + d_raster.color_alpha[3] * (g.color_alpha[3] - base.color_alpha[3])
    };

    let mut temp = scene.clone();
    for axis in 0..3 {
        let idx = splat * 3 + axis;
        temp.positions[idx] += eps;
        let plus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
        temp.positions[idx] -= 2.0 * eps;
        let minus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
        temp.positions[idx] += eps;
        d_pos[axis] = (loss(&plus) - loss(&minus)) / (2.0 * eps);
    }
    for axis in 0..3 {
        let idx = splat * 3 + axis;
        temp.scales[idx] += eps;
        let plus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
        temp.scales[idx] -= 2.0 * eps;
        let minus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
        temp.scales[idx] += eps;
        d_scale[axis] = (loss(&plus) - loss(&minus)) / (2.0 * eps);
    }
    for axis in 0..4 {
        let idx = splat * 4 + axis;
        temp.rotations[idx] += eps;
        let plus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
        temp.rotations[idx] -= 2.0 * eps;
        let minus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
        temp.rotations[idx] += eps;
        d_rot[axis] = (loss(&plus) - loss(&minus)) / (2.0 * eps);
    }
    temp.opacities[splat] += eps;
    let plus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
    temp.opacities[splat] -= 2.0 * eps;
    let minus = build_raster_gaussian(&temp, splat, camera, radius_scale, max_anisotropy, sh_band);
    d_raw_opacity = (loss(&plus) - loss(&minus)) / (2.0 * eps);

    (d_pos, d_scale, d_rot, d_raw_opacity)
}
