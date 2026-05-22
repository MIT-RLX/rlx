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
use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::core::Camera;

pub const DEFAULT_COLMAP_IMPORT_MIN_TRACK_LENGTH: i32 = 3;

#[derive(Clone, Debug)]
pub struct ColmapCamera {
    pub camera_id: i32,
    pub model_id: i32,
    pub width: u64,
    pub height: u64,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub k1: f64,
    pub k2: f64,
    pub p1: f64,
    pub p2: f64,
    pub k3: f64,
    pub k4: f64,
    pub k5: f64,
    pub k6: f64,
}

#[derive(Clone, Debug)]
pub struct ColmapImage {
    pub image_id: i32,
    pub q_wxyz: [f32; 4],
    pub t_xyz: [f32; 3],
    pub camera_id: i32,
    pub name: String,
    pub points2d_xy: Vec<[f32; 2]>,
    pub points2d_point3d_ids: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct ColmapPoint3D {
    pub point_id: u64,
    pub xyz: [f32; 3],
    pub rgb: [f32; 3],
    pub error: f64,
    pub track_length: u64,
}

#[derive(Clone, Debug)]
pub struct ColmapReconstruction {
    pub root: PathBuf,
    pub sparse_dir: PathBuf,
    pub cameras: BTreeMap<i32, ColmapCamera>,
    pub images: BTreeMap<i32, ColmapImage>,
    pub points3d: BTreeMap<u64, ColmapPoint3D>,
}

#[derive(Clone, Debug)]
pub struct ColmapFrame {
    pub image_id: i32,
    pub image_path: PathBuf,
    pub q_wxyz: [f32; 4],
    pub t_xyz: [f32; 3],
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub width: u32,
    pub height: u32,
    pub k1: f32,
    pub k2: f32,
    pub p1: f32,
    pub p2: f32,
    pub k3: f32,
    pub k4: f32,
    pub k5: f32,
    pub k6: f32,
    pub camera_id: i32,
}

impl ColmapFrame {
    pub fn make_camera(&self, near: f32, far: f32) -> Camera {
        Camera::from_colmap(
            self.q_wxyz,
            self.t_xyz,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            [
                Some(self.k1),
                Some(self.k2),
                Some(self.p1),
                Some(self.p2),
                Some(self.k3),
                Some(self.k4),
                Some(self.k5),
                Some(self.k6),
            ],
            near,
            far,
        )
    }
}

#[derive(Clone, Debug, Default)]
pub struct GaussianInitHyperParams {
    pub position_jitter_std: Option<f32>,
    pub base_scale: Option<f32>,
    pub scale_jitter_ratio: Option<f32>,
    pub initial_opacity: Option<f32>,
    pub color_jitter_std: Option<f32>,
}

pub fn point_tables(recon: &ColmapReconstruction, min_track_length: i32) -> (Vec<f32>, Vec<f32>) {
    let min_track = min_track_length.max(0) as u64;
    if recon.points3d.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let points: Vec<&ColmapPoint3D> = recon
        .points3d
        .values()
        .filter(|p| p.track_length >= min_track)
        .collect();
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut xyz = Vec::with_capacity(points.len() * 3);
    let mut rgb = Vec::with_capacity(points.len() * 3);
    for point in points {
        xyz.extend_from_slice(&point.xyz);
        rgb.extend_from_slice(&point.rgb);
    }
    (xyz, rgb)
}

pub fn colmap_camera_centers(recon: &ColmapReconstruction) -> Vec<[f32; 3]> {
    use crate::core::math::{mat3_transpose_mul_vec3, rotation_matrix_from_quaternion_wxyz};
    recon
        .images
        .values()
        .map(|image| {
            let rot = rotation_matrix_from_quaternion_wxyz(image.q_wxyz);
            mat3_transpose_mul_vec3(rot, [-image.t_xyz[0], -image.t_xyz[1], -image.t_xyz[2]])
        })
        .collect()
}
