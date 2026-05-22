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
//! Camera model aligned with `src/renderer/camera.py`.

use super::constants::{DISTORTION_EPS, DISTORTION_NEWTON_ITERS, VEC_EPS};
use super::math::{cross, dot3, mat3_mul_vec3, mat3_transpose_mul_vec3, normalize3};

#[derive(Clone, Debug)]
pub struct Camera {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub up: [f32; 3],
    pub fov_y_degrees: f32,
    pub near: f32,
    pub far: f32,
    pub min_camera_distance: f32,
    pub fx: Option<f32>,
    pub fy: Option<f32>,
    pub cx: Option<f32>,
    pub cy: Option<f32>,
    pub distortion: [Option<f32>; 8],
    pub basis_override: Option<[[f32; 3]; 3]>,
}

impl Camera {
    pub fn look_at(
        position: [f32; 3],
        target: [f32; 3],
        up: [f32; 3],
        fov_y_degrees: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            target,
            up: normalize3(up, VEC_EPS),
            fov_y_degrees,
            near,
            far,
            min_camera_distance: 0.0,
            fx: None,
            fy: None,
            cx: None,
            cy: None,
            distortion: [None; 8],
            basis_override: None,
        }
    }

    pub fn focal_pixels(&self, height: u32) -> f32 {
        self.fy
            .unwrap_or_else(|| 0.5 * height as f32 / (0.5 * self.fov_y_degrees.to_radians()).tan())
    }

    pub fn focal_pixels_xy(&self, width: u32, height: u32) -> (f32, f32) {
        let focal_y = self.focal_pixels(height);
        (
            self.fx.unwrap_or(focal_y),
            self.fy.unwrap_or(focal_y),
        )
    }

    pub fn principal_point(&self, width: u32, height: u32) -> (f32, f32) {
        (
            self.cx.unwrap_or(0.5 * width as f32),
            self.cy.unwrap_or(0.5 * height as f32),
        )
    }

    pub fn distortion_params(&self) -> [f64; 8] {
        let defaults = [0.0; 8];
        std::array::from_fn(|i| self.distortion[i].unwrap_or(defaults[i]) as f64)
    }

    pub fn basis(&self) -> [[f32; 3]; 3] {
        if let Some(basis) = self.basis_override {
            return [
                normalize3(basis[0], VEC_EPS),
                normalize3(basis[1], VEC_EPS),
                normalize3(basis[2], VEC_EPS),
            ];
        }
        let forward = normalize3(
            [
                self.target[0] - self.position[0],
                self.target[1] - self.position[1],
                self.target[2] - self.position[2],
            ],
            VEC_EPS,
        );
        let right = normalize3(cross(self.up, forward), VEC_EPS);
        let up = normalize3(cross(forward, right), VEC_EPS);
        [right, up, forward]
    }

    pub fn world_to_camera(&self, world_vector: [f32; 3]) -> [f32; 3] {
        mat3_mul_vec3(self.basis(), world_vector)
    }

    pub fn camera_to_world(&self, camera_vector: [f32; 3]) -> [f32; 3] {
        mat3_transpose_mul_vec3(self.basis(), camera_vector)
    }

    pub fn world_point_to_camera(&self, world_pos: [f32; 3]) -> [f32; 3] {
        self.world_to_camera([
            world_pos[0] - self.position[0],
            world_pos[1] - self.position[1],
            world_pos[2] - self.position[2],
        ])
    }

    pub fn camera_point_to_world(&self, camera_pos: [f32; 3]) -> [f32; 3] {
        let offset = self.camera_to_world(camera_pos);
        [
            self.position[0] + offset[0],
            self.position[1] + offset[1],
            self.position[2] + offset[2],
        ]
    }

    pub fn project_camera_to_screen(
        &self,
        camera_pos: [f32; 3],
        width: u32,
        height: u32,
    ) -> ([f32; 2], bool) {
        let depth = camera_pos[2];
        if !depth.is_finite() || depth <= 1e-12 {
            return ([0.0, 0.0], false);
        }
        let (fx, fy) = self.focal_pixels_xy(width, height);
        let (cx, cy) = self.principal_point(width, height);
        let uv = Self::distort_normalized(
            [camera_pos[0] as f64 / depth as f64, camera_pos[1] as f64 / depth as f64],
            self.distortion_params(),
        );
        let screen = [uv[0] * fx + cx, uv[1] * fy + cy];
        (screen, screen[0].is_finite() && screen[1].is_finite())
    }

    pub fn project_world_to_screen(
        &self,
        world_pos: [f32; 3],
        width: u32,
        height: u32,
    ) -> ([f32; 2], bool) {
        self.project_camera_to_screen(self.world_point_to_camera(world_pos), width, height)
    }

    pub fn screen_to_world(
        &self,
        screen_pos: [f32; 2],
        depth: f32,
        width: u32,
        height: u32,
    ) -> [f32; 3] {
        let (fx, fy) = self.focal_pixels_xy(width, height);
        let (cx, cy) = self.principal_point(width, height);
        let uv = [
            (screen_pos[0] - cx) as f64 / fx.max(1e-12) as f64,
            (screen_pos[1] - cy) as f64 / fy.max(1e-12) as f64,
        ];
        let undistorted = Self::undistort_normalized(uv, self.distortion_params());
        let depth_safe = depth.max(1e-12);
        self.camera_point_to_world([
            (undistorted[0] * depth_safe as f64) as f32,
            (undistorted[1] * depth_safe as f64) as f32,
            depth_safe,
        ])
    }

    pub fn screen_to_world_ray(&self, screen_pos: [f32; 2], width: u32, height: u32) -> [f32; 3] {
        let world = self.screen_to_world(screen_pos, 1.0, width, height);
        normalize3(
            [
                world[0] - self.position[0],
                world[1] - self.position[1],
                world[2] - self.position[2],
            ],
            VEC_EPS,
        )
    }

    /// Build a camera from COLMAP world-to-camera quaternion and translation.
    pub fn from_colmap(
        q_wxyz: [f32; 4],
        t_xyz: [f32; 3],
        fx: f32,
        fy: f32,
        cx: f32,
        cy: f32,
        distortion: [Option<f32>; 8],
        near: f32,
        far: f32,
    ) -> Self {
        use super::math::{mat3_transpose_mul_vec3, rotation_matrix_from_quaternion_wxyz};
        let rot = rotation_matrix_from_quaternion_wxyz(q_wxyz);
        let cam_pos = mat3_transpose_mul_vec3(rot, [-t_xyz[0], -t_xyz[1], -t_xyz[2]]);
        let forward = normalize3(rot[2], VEC_EPS);
        let up = normalize3(rot[1], VEC_EPS);
        let target = [
            cam_pos[0] + forward[0],
            cam_pos[1] + forward[1],
            cam_pos[2] + forward[2],
        ];
        Self {
            position: cam_pos,
            target,
            up,
            fov_y_degrees: 60.0,
            near,
            far,
            min_camera_distance: 0.0,
            fx: Some(fx),
            fy: Some(fy),
            cx: Some(cx),
            cy: Some(cy),
            distortion,
            basis_override: Some(rot),
        }
    }

    fn safe_denominator(value: f64) -> f64 {
        if value.abs() > DISTORTION_EPS {
            value
        } else if value != 0.0 {
            value.signum() * DISTORTION_EPS
        } else {
            DISTORTION_EPS
        }
    }

    fn radial_distortion(r2: f64, params: [f64; 8]) -> (f64, f64) {
        let (k1, k2, _, _, k3, k4, k5, k6) = (
            params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
        );
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let denominator = Self::safe_denominator(1.0 + k4 * r2 + k5 * r4 + k6 * r6);
        let radial = numerator / denominator;
        let d_numerator = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
        let d_denominator = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
        let d_radial_dr2 =
            (d_numerator * denominator - numerator * d_denominator) / (denominator * denominator);
        (radial, d_radial_dr2)
    }

    fn tangential_distortion(x: f64, y: f64, r2: f64, p1: f64, p2: f64) -> (f64, f64) {
        (
            2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
            p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
        )
    }

    fn distort_normalized_with_params(uv: [f64; 2], params: [f64; 8]) -> [f64; 2] {
        let (x, y) = (uv[0], uv[1]);
        let (k1, k2, p1, p2, k3, k4, k5, k6) = (
            params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
        );
        let r2 = x * x + y * y;
        let (radial, _) = Self::radial_distortion(r2, params);
        let (tx, ty) = Self::tangential_distortion(x, y, r2, p1, p2);
        let _ = (k1, k2, k3, k4, k5, k6);
        [x * radial + tx, y * radial + ty]
    }

    fn distort_normalized(uv: [f64; 2], params: [f64; 8]) -> [f32; 2] {
        let out = Self::distort_normalized_with_params(uv, params);
        [out[0] as f32, out[1] as f32]
    }

    fn distortion_jacobian(uv: [f64; 2], params: [f64; 8]) -> [[f64; 2]; 2] {
        let (x, y) = (uv[0], uv[1]);
        let (p1, p2) = (params[2], params[3]);
        let r2 = x * x + y * y;
        let (radial, d_radial_dr2) = Self::radial_distortion(r2, params);
        let d_radial_dx = d_radial_dr2 * 2.0 * x;
        let d_radial_dy = d_radial_dr2 * 2.0 * y;
        [
            [
                radial + x * d_radial_dx + 2.0 * p1 * y + 6.0 * p2 * x,
                x * d_radial_dy + 2.0 * p1 * x + 2.0 * p2 * y,
            ],
            [
                y * d_radial_dx + 2.0 * p1 * x + 2.0 * p2 * y,
                radial + y * d_radial_dy + 6.0 * p1 * y + 2.0 * p2 * x,
            ],
        ]
    }

    fn undistort_normalized(uv_distorted: [f64; 2], params: [f64; 8]) -> [f64; 2] {
        if !uv_distorted[0].is_finite() || !uv_distorted[1].is_finite() {
            return uv_distorted;
        }
        if params.iter().all(|v| v.abs() <= DISTORTION_EPS) {
            return uv_distorted;
        }
        let mut estimate = uv_distorted;
        for _ in 0..DISTORTION_NEWTON_ITERS {
            let error = {
                let distorted = Self::distort_normalized_with_params(estimate, params);
                [distorted[0] - uv_distorted[0], distorted[1] - uv_distorted[1]]
            };
            if dot3([error[0] as f32, error[1] as f32, 0.0], [error[0] as f32, error[1] as f32, 0.0])
                <= (DISTORTION_EPS * DISTORTION_EPS) as f32
            {
                break;
            }
            let jacobian = Self::distortion_jacobian(estimate, params);
            let det = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
            if !det.is_finite() || det.abs() <= DISTORTION_EPS {
                break;
            }
            let step = [
                (jacobian[1][1] * error[0] - jacobian[0][1] * error[1]) / det,
                (-jacobian[1][0] * error[0] + jacobian[0][0] * error[1]) / det,
            ];
            let next = [estimate[0] - step[0], estimate[1] - step[1]];
            if !next[0].is_finite() || !next[1].is_finite() {
                break;
            }
            estimate = next;
        }
        estimate
    }
}
