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
use crate::core::constants::VEC_EPS;
use crate::core::math::{
    mat3_transpose_mul_vec3, normalize3, quaternion_from_rotation_matrix,
    rotation_matrix_from_quaternion_wxyz,
};

use super::types::{ColmapImage, ColmapReconstruction};

type Mat4 = [[f32; 4]; 4];

pub fn transform_poses_pca(poses: &[Mat4], rescale: bool) -> (Vec<Mat4>, Mat4) {
    if poses.is_empty() {
        return (Vec::new(), identity4());
    }
    let colmap2opengl = diag4([1.0, -1.0, -1.0, 1.0]);
    let mut transformed: Vec<Mat4> = poses.iter().map(|p| mat4_mul(colmap2opengl, *p)).collect();

    let count = transformed.len() as f32;
    let mut mean = [0.0f32; 3];
    for pose in &transformed {
        mean[0] += pose[0][3];
        mean[1] += pose[1][3];
        mean[2] += pose[2][3];
    }
    for v in &mut mean {
        *v /= count;
    }

    let mut cov = [[0.0f32; 3]; 3];
    for pose in &transformed {
        let d = [
            pose[0][3] - mean[0],
            pose[1][3] - mean[1],
            pose[2][3] - mean[2],
        ];
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += d[i] * d[j];
            }
        }
    }

    let (eigvecs, _) = eigen_symmetric_3x3(cov);
    let mut rotation = transpose3(eigvecs);
    if det3(rotation) < 0.0 {
        rotation[2][0] *= -1.0;
        rotation[2][1] *= -1.0;
        rotation[2][2] *= -1.0;
    }

    let mut transform = identity4();
    for i in 0..3 {
        for j in 0..3 {
            transform[i][j] = rotation[i][j];
        }
        transform[i][3] =
            rotation[i][0] * (-mean[0]) + rotation[i][1] * (-mean[1]) + rotation[i][2] * (-mean[2]);
    }
    transformed = transformed
        .into_iter()
        .map(|p| mat4_mul(transform, p))
        .collect();

    let mean_y2 = transformed.iter().map(|p| p[2][1]).sum::<f32>() / count;
    if mean_y2 < 0.0 {
        let flip = diag4([1.0, -1.0, -1.0, 1.0]);
        transform = mat4_mul(flip, transform);
        transformed = transformed.into_iter().map(|p| mat4_mul(flip, p)).collect();
    }

    if rescale {
        let (scaled, scale_xform) = rescale_poses_to_unit_cube(&transformed, transform);
        transformed = scaled;
        transform = scale_xform;
    }

    let aligned2colmap: Mat4 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    transform = mat4_mul(aligned2colmap, transform);
    transformed = transformed
        .into_iter()
        .map(|p| mat4_mul(aligned2colmap, p))
        .collect();
    let inv_colmap2opengl = diag4([1.0, -1.0, -1.0, 1.0]);
    transformed = transformed
        .into_iter()
        .map(|p| mat4_mul(p, inv_colmap2opengl))
        .collect();
    (transformed, transform)
}

pub fn transform_colmap_reconstruction_pca(
    recon: &ColmapReconstruction,
    rescale: bool,
) -> (ColmapReconstruction, Mat4) {
    let image_items: Vec<(i32, ColmapImage)> =
        recon.images.iter().map(|(k, v)| (*k, v.clone())).collect();
    if image_items.is_empty() {
        return (recon.clone(), identity4());
    }
    let poses: Vec<Mat4> = image_items
        .iter()
        .map(|(_, image)| camera_to_world_pose(image.q_wxyz, image.t_xyz))
        .collect();
    let (transformed_poses, transform) = transform_poses_pca(&poses, rescale);

    let mut transformed_images = recon.images.clone();
    for ((image_id, _), pose) in image_items.iter().zip(transformed_poses.iter()) {
        let (rot_wc, t) = world_to_camera_from_pose(pose);
        let q = quaternion_from_rotation_matrix(rot_wc);
        if let Some(entry) = transformed_images.get_mut(image_id) {
            entry.q_wxyz = q;
            entry.t_xyz = t;
        }
    }

    let mut transformed_points = recon.points3d.clone();
    for point in transformed_points.values_mut() {
        point.xyz = transform_point3(transform, point.xyz);
    }

    (
        ColmapReconstruction {
            root: recon.root.clone(),
            sparse_dir: recon.sparse_dir.clone(),
            cameras: recon.cameras.clone(),
            images: transformed_images,
            points3d: transformed_points,
        },
        transform,
    )
}

pub(crate) fn camera_to_world_pose(q_wxyz: [f32; 4], t_xyz: [f32; 3]) -> Mat4 {
    let rot_wc = rotation_matrix_from_quaternion_wxyz(q_wxyz);
    let rot_cw = transpose3(rot_wc);
    let center = mat3_transpose_mul_vec3(rot_wc, [-t_xyz[0], -t_xyz[1], -t_xyz[2]]);
    let mut pose = identity4();
    for i in 0..3 {
        for j in 0..3 {
            pose[i][j] = rot_cw[i][j];
        }
        pose[i][3] = center[i];
    }
    pose
}

fn world_to_camera_from_pose(pose: &Mat4) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut rot_cw = [
        [pose[0][0], pose[0][1], pose[0][2]],
        [pose[1][0], pose[1][1], pose[1][2]],
        [pose[2][0], pose[2][1], pose[2][2]],
    ];
    rot_cw = orthonormalize_rotation(rot_cw);
    let center = [pose[0][3], pose[1][3], pose[2][3]];
    let rot_wc = transpose3(rot_cw);
    let t = [
        -(rot_wc[0][0] * center[0] + rot_wc[0][1] * center[1] + rot_wc[0][2] * center[2]),
        -(rot_wc[1][0] * center[0] + rot_wc[1][1] * center[1] + rot_wc[1][2] * center[2]),
        -(rot_wc[2][0] * center[0] + rot_wc[2][1] * center[1] + rot_wc[2][2] * center[2]),
    ];
    (rot_wc, t)
}

fn transform_point3(transform: Mat4, xyz: [f32; 3]) -> [f32; 3] {
    [
        transform[0][0] * xyz[0]
            + transform[0][1] * xyz[1]
            + transform[0][2] * xyz[2]
            + transform[0][3],
        transform[1][0] * xyz[0]
            + transform[1][1] * xyz[1]
            + transform[1][2] * xyz[2]
            + transform[1][3],
        transform[2][0] * xyz[0]
            + transform[2][1] * xyz[1]
            + transform[2][2] * xyz[2]
            + transform[2][3],
    ]
}

fn rescale_poses_to_unit_cube(poses: &[Mat4], transform: Mat4) -> (Vec<Mat4>, Mat4) {
    let mut max_extent = 0.0f32;
    for pose in poses {
        for axis in 0..3 {
            max_extent = max_extent.max(pose[axis][3].abs());
        }
    }
    if !max_extent.is_finite() || max_extent <= 1e-8 {
        return (poses.to_vec(), transform);
    }
    let scale = 1.0 / max_extent;
    let mut scale_transform = identity4();
    scale_transform[0][0] = scale;
    scale_transform[1][1] = scale;
    scale_transform[2][2] = scale;
    let scaled = poses
        .iter()
        .map(|p| mat4_mul(scale_transform, *p))
        .collect();
    (scaled, mat4_mul(scale_transform, transform))
}

fn identity4() -> Mat4 {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn diag4(d: [f32; 4]) -> Mat4 {
    [
        [d[0], 0.0, 0.0, 0.0],
        [0.0, d[1], 0.0, 0.0],
        [0.0, 0.0, d[2], 0.0],
        [0.0, 0.0, 0.0, d[3]],
    ]
}

fn mat4_mul(a: Mat4, b: Mat4) -> Mat4 {
    let mut out = identity4();
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] =
                a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    out
}

fn transpose3(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        [m[0][0], m[1][0], m[2][0]],
        [m[0][1], m[1][1], m[2][1]],
        [m[0][2], m[1][2], m[2][2]],
    ]
}

fn det3(m: [[f32; 3]; 3]) -> f32 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

fn orthonormalize_rotation(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut u = [m[0], m[1], m[2]];
    u[0] = normalize3(u[0], VEC_EPS);
    let dot01 = u[1][0] * u[0][0] + u[1][1] * u[0][1] + u[1][2] * u[0][2];
    u[1] = [
        u[1][0] - dot01 * u[0][0],
        u[1][1] - dot01 * u[0][1],
        u[1][2] - dot01 * u[0][2],
    ];
    u[1] = normalize3(u[1], VEC_EPS);
    u[2] = normalize3(
        [
            u[0][1] * u[1][2] - u[0][2] * u[1][1],
            u[0][2] * u[1][0] - u[0][0] * u[1][2],
            u[0][0] * u[1][1] - u[0][1] * u[1][0],
        ],
        VEC_EPS,
    );
    u
}

fn eigen_symmetric_3x3(a: [[f32; 3]; 3]) -> ([[f32; 3]; 3], [f32; 3]) {
    let mut b = a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..16 {
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max = b[0][1].abs();
        if b[0][2].abs() > max {
            max = b[0][2].abs();
            p = 0;
            q = 2;
        }
        if b[1][2].abs() > max {
            p = 1;
            q = 2;
        }
        if max < 1e-12 {
            break;
        }
        let theta = 0.5 * (b[q][q] - b[p][p]).atan2(2.0 * b[p][q]);
        let c = theta.cos();
        let s = theta.sin();
        jacobi_rotate(&mut b, &mut v, p, q, c, s);
    }
    ([v[0], v[1], v[2]], [b[0][0], b[1][1], b[2][2]])
}

fn jacobi_rotate(b: &mut [[f32; 3]; 3], v: &mut [[f32; 3]; 3], p: usize, q: usize, c: f32, s: f32) {
    for i in 0..3 {
        let bip = b[i][p];
        let biq = b[i][q];
        b[i][p] = c * bip - s * biq;
        b[i][q] = s * bip + c * biq;
    }
    for i in 0..3 {
        let bpi = b[p][i];
        let bqi = b[q][i];
        b[p][i] = c * bpi - s * bqi;
        b[q][i] = s * bpi + c * bqi;
    }
    for i in 0..3 {
        let vip = v[i][p];
        let viq = v[i][q];
        v[i][p] = c * vip - s * viq;
        v[i][q] = s * vip + c * viq;
    }
}
