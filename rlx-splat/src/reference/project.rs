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
use crate::core::{
    ALPHA_CUTOFF_DEFAULT, Camera, ELLIPSE_EPS, ELLIPSE_RADIUS_PAD_PX,
    GAUSSIAN_SUPPORT_SIGMA_RADIUS, GaussianScene, quat_conj, quat_rotate,
};
use crate::core::{evaluate_sh0_sh1, resolve_supported_sh_coeffs};

#[derive(Clone, Debug)]
pub struct ProjectedSplats {
    pub center_radius_depth: Vec<f32>,
    pub ellipse_conic: Vec<f32>,
    pub color_alpha: Vec<f32>,
    pub opacity_scale: Vec<f32>,
    pub valid: Vec<u32>,
    pub pos_local: Vec<f32>,
    pub inv_scale: Vec<f32>,
    pub quat: Vec<f32>,
}

fn init_fullscreen_fallback_ellipse(width: u32, height: u32) -> ([f32; 2], f32, [f32; 3]) {
    let radius = (width.max(height).max(1)) as f32;
    let inv_radius_sq = 1.0 / (radius * radius).max(ELLIPSE_EPS);
    let center = [0.5 * width as f32, 0.5 * height as f32];
    (center, radius, [inv_radius_sq, 0.0, inv_radius_sq])
}

fn solve_conic_renorm(points: &[[f32; 2]; 5], eps: f64) -> Option<[f32; 5]> {
    let sx = (points[0][0] - points[1][0]) as f64;
    let sy = (points[1][1] - points[0][1]) as f64;
    if sx.abs() <= eps || sy.abs() <= eps {
        return None;
    }
    let inv_sx = 1.0 / sx;
    let inv_sy = 1.0 / sy;
    let offset_x = points[1][0] as f64;
    let offset_y = points[0][1] as f64;
    let mut uv = [[0.0f64; 2]; 3];
    for i in 0..3 {
        uv[i][0] = (points[2 + i][0] as f64 - offset_x) * inv_sx;
        uv[i][1] = (points[2 + i][1] as f64 - offset_y) * inv_sy;
    }
    let m00 = uv[0][0] * uv[0][0] - uv[0][0];
    let mut m01 = 2.0 * uv[0][0] * uv[0][1];
    let mut m02 = uv[0][1] * uv[0][1] - uv[0][1];
    let mut r0 = uv[0][0] + uv[0][1] - 1.0;
    let m10 = uv[1][0] * uv[1][0] - uv[1][0];
    let mut m11 = 2.0 * uv[1][0] * uv[1][1];
    let mut m12 = uv[1][1] * uv[1][1] - uv[1][1];
    let mut r1 = uv[1][0] + uv[1][1] - 1.0;
    let m20 = uv[2][0] * uv[2][0] - uv[2][0];
    let mut m21 = 2.0 * uv[2][0] * uv[2][1];
    let mut m22 = uv[2][1] * uv[2][1] - uv[2][1];
    let mut r2 = uv[2][0] + uv[2][1] - 1.0;
    if m00.abs() <= eps {
        return None;
    }
    let inv_m00 = 1.0 / m00;
    m01 *= inv_m00;
    m02 *= inv_m00;
    r0 *= inv_m00;
    let factor = m10;
    m11 -= factor * m01;
    m12 -= factor * m02;
    r1 -= factor * r0;
    let factor = m20;
    m21 -= factor * m01;
    m22 -= factor * m02;
    r2 -= factor * r0;
    if m11.abs() <= eps {
        return None;
    }
    let inv_m11 = 1.0 / m11;
    m12 *= inv_m11;
    r1 *= inv_m11;
    let factor = m21;
    m22 -= factor * m12;
    r2 -= factor * r1;
    if m22.abs() <= eps {
        return None;
    }
    let conic_c = r2 / m22;
    let conic_b = r1 - m12 * conic_c;
    let conic_a = r0 - m01 * conic_b - m02 * conic_c;
    let conic_d = -(conic_a + 1.0);
    let conic_e = -(conic_c + 1.0);
    let inv_sx2 = inv_sx * inv_sx;
    let inv_sy2 = inv_sy * inv_sy;
    let inv_sx_sy = inv_sx * inv_sy;
    let coeff_a = conic_a * inv_sx2;
    let coeff_b = conic_b * inv_sx_sy;
    let coeff_c = conic_c * inv_sy2;
    let coeff_d = -2.0 * coeff_a * offset_x - 2.0 * coeff_b * offset_y + conic_d * inv_sx;
    let coeff_e = -2.0 * coeff_b * offset_x - 2.0 * coeff_c * offset_y + conic_e * inv_sy;
    let coeff_f = coeff_a * offset_x * offset_x
        + 2.0 * coeff_b * offset_x * offset_y
        + coeff_c * offset_y * offset_y
        - conic_d * inv_sx * offset_x
        - conic_e * inv_sy * offset_y
        + 1.0;
    if coeff_f.abs() <= eps {
        return None;
    }
    let inv_f = -1.0 / coeff_f;
    Some([
        (coeff_a * inv_f) as f32,
        (coeff_b * inv_f) as f32,
        (coeff_c * inv_f) as f32,
        (coeff_d * inv_f) as f32,
        (coeff_e * inv_f) as f32,
    ])
}

fn support_sphere_intersects_view_frustum(
    camera_center: [f32; 3],
    camera: &Camera,
    width: u32,
    height: u32,
    support_radius: f32,
) -> bool {
    if !camera_center[0].is_finite()
        || !camera_center[1].is_finite()
        || !camera_center[2].is_finite()
        || !support_radius.is_finite()
        || support_radius <= 0.0
    {
        return false;
    }
    if camera_center[2] + support_radius <= 1e-4 {
        return false;
    }
    if camera_center[2] - support_radius >= camera.far {
        return false;
    }
    let (fx, fy) = camera.focal_pixels_xy(width, height);
    let (cx, cy) = camera.principal_point(width, height);
    let w = width as f32;
    let h = height as f32;
    let planes = [
        [fx, 0.0, cx],
        [-fx, 0.0, w - cx],
        [0.0, fy, cy],
        [0.0, -fy, h - cy],
    ];
    for plane in planes {
        let dot =
            plane[0] * camera_center[0] + plane[1] * camera_center[1] + plane[2] * camera_center[2];
        let norm = (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]).sqrt();
        if dot < -support_radius * norm {
            return false;
        }
    }
    true
}

fn compute_outline_ellipse(
    world_pos: [f32; 3],
    inv_scale: [f32; 3],
    rotation: [f32; 4],
    camera: &Camera,
    width: u32,
    height: u32,
) -> Option<([f32; 2], f32, [f32; 3])> {
    let (screen_center, screen_ok) = camera.project_world_to_screen(world_pos, width, height);
    let scale = [
        1.0 / inv_scale[0].max(1e-12),
        1.0 / inv_scale[1].max(1e-12),
        1.0 / inv_scale[2].max(1e-12),
    ];
    let camera_center = camera.world_point_to_camera(world_pos);
    let support_radius = scale[0].max(scale[1]).max(scale[2]);
    let support_intersects = support_sphere_intersects_view_frustum(
        camera_center,
        camera,
        width,
        height,
        support_radius,
    );
    if !screen_ok && !support_intersects {
        return None;
    }
    let view_origin_local = {
        let delta = [
            camera.position[0] - world_pos[0],
            camera.position[1] - world_pos[1],
            camera.position[2] - world_pos[2],
        ];
        let rotated = quat_rotate(delta, rotation);
        [
            rotated[0] * inv_scale[0],
            rotated[1] * inv_scale[1],
            rotated[2] * inv_scale[2],
        ]
    };
    let view_distance = (view_origin_local[0] * view_origin_local[0]
        + view_origin_local[1] * view_origin_local[1]
        + view_origin_local[2] * view_origin_local[2])
        .sqrt();
    if view_distance <= 1.0 + ELLIPSE_EPS {
        return if support_intersects {
            Some(init_fullscreen_fallback_ellipse(width, height))
        } else {
            None
        };
    }
    let view_dir_local = [
        view_origin_local[0] / view_distance,
        view_origin_local[1] / view_distance,
        view_origin_local[2] / view_distance,
    ];
    let tangent_circle_center = [
        view_dir_local[0] / view_distance,
        view_dir_local[1] / view_distance,
        view_dir_local[2] / view_distance,
    ];
    let tangent_circle_radius = (1.0 - 1.0 / (view_distance * view_distance))
        .max(0.0)
        .sqrt();
    let tangent_axis = if view_dir_local[2].abs() < 0.999 {
        [0.0, 0.0, 1.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let mut tangent_basis_u = crate::core::cross(tangent_axis, view_dir_local);
    let u_norm = (tangent_basis_u[0] * tangent_basis_u[0]
        + tangent_basis_u[1] * tangent_basis_u[1]
        + tangent_basis_u[2] * tangent_basis_u[2])
        .sqrt();
    if u_norm <= ELLIPSE_EPS || !u_norm.is_finite() {
        return None;
    }
    tangent_basis_u = [
        tangent_basis_u[0] / u_norm,
        tangent_basis_u[1] / u_norm,
        tangent_basis_u[2] / u_norm,
    ];
    let mut tangent_basis_v = crate::core::cross(view_dir_local, tangent_basis_u);
    let v_norm = (tangent_basis_v[0] * tangent_basis_v[0]
        + tangent_basis_v[1] * tangent_basis_v[1]
        + tangent_basis_v[2] * tangent_basis_v[2])
        .sqrt();
    if v_norm <= ELLIPSE_EPS || !v_norm.is_finite() {
        return None;
    }
    tangent_basis_v = [
        tangent_basis_v[0] / v_norm,
        tangent_basis_v[1] / v_norm,
        tangent_basis_v[2] / v_norm,
    ];
    let mut outline_points = [[0.0f32; 2]; 5];
    let mut outline_min = [1e30f32; 2];
    let mut outline_max = [-1e30f32; 2];
    let q_inv = quat_conj(rotation);
    for index in 0..5 {
        let theta = 2.0 * std::f32::consts::PI * index as f32 / 5.0;
        let local_point = [
            tangent_circle_center[0]
                + tangent_circle_radius
                    * (theta.cos() * tangent_basis_u[0] + theta.sin() * tangent_basis_v[0]),
            tangent_circle_center[1]
                + tangent_circle_radius
                    * (theta.cos() * tangent_basis_u[1] + theta.sin() * tangent_basis_v[1]),
            tangent_circle_center[2]
                + tangent_circle_radius
                    * (theta.cos() * tangent_basis_u[2] + theta.sin() * tangent_basis_v[2]),
        ];
        let scaled = [
            local_point[0] * scale[0],
            local_point[1] * scale[1],
            local_point[2] * scale[2],
        ];
        let world_point = {
            let rotated = quat_rotate(scaled, q_inv);
            [
                world_pos[0] + rotated[0],
                world_pos[1] + rotated[1],
                world_pos[2] + rotated[2],
            ]
        };
        let (screen_point, point_ok) = camera.project_world_to_screen(world_point, width, height);
        if !point_ok {
            return if support_intersects {
                Some(init_fullscreen_fallback_ellipse(width, height))
            } else {
                None
            };
        }
        outline_points[index] = screen_point;
        outline_min[0] = outline_min[0].min(screen_point[0]);
        outline_min[1] = outline_min[1].min(screen_point[1]);
        outline_max[0] = outline_max[0].max(screen_point[0]);
        outline_max[1] = outline_max[1].max(screen_point[1]);
    }
    if screen_ok {
        outline_min[0] = outline_min[0].min(screen_center[0]);
        outline_min[1] = outline_min[1].min(screen_center[1]);
        outline_max[0] = outline_max[0].max(screen_center[0]);
        outline_max[1] = outline_max[1].max(screen_center[1]);
    }
    if outline_max[0] < 0.0
        || outline_min[0] >= width as f32
        || outline_max[1] < 0.0
        || outline_min[1] >= height as f32
    {
        return None;
    }
    let bbox_center = [
        0.5 * (outline_min[0] + outline_max[0]),
        0.5 * (outline_min[1] + outline_max[1]),
    ];
    let bbox_half_extent = [
        (0.5 * (outline_max[0] - outline_min[0])).max(ELLIPSE_EPS),
        (0.5 * (outline_max[1] - outline_min[1])).max(ELLIPSE_EPS),
    ];
    let mut norm_points = [[0.0f32; 2]; 5];
    for i in 0..5 {
        norm_points[i] = [
            (outline_points[i][0] - bbox_center[0]) / bbox_half_extent[0],
            (outline_points[i][1] - bbox_center[1]) / bbox_half_extent[1],
        ];
    }
    let solution = solve_conic_renorm(&norm_points, ELLIPSE_EPS as f64)?;
    let conic_norm = [solution[0] as f64, solution[1] as f64, solution[2] as f64];
    let linear = [solution[3] as f64, solution[4] as f64];
    let mut det_a = conic_norm[0] * conic_norm[2] - conic_norm[1] * conic_norm[1];
    if det_a <= ELLIPSE_EPS as f64 {
        return None;
    }
    let ellipse_center = [
        0.5 * (conic_norm[1] * linear[1] - conic_norm[2] * linear[0]) / det_a,
        0.5 * (conic_norm[1] * linear[0] - conic_norm[0] * linear[1]) / det_a,
    ];
    let center_times_a = [
        conic_norm[0] * ellipse_center[0] + conic_norm[1] * ellipse_center[1],
        conic_norm[1] * ellipse_center[0] + conic_norm[2] * ellipse_center[1],
    ];
    let center_scale =
        1.0 + ellipse_center[0] * center_times_a[0] + ellipse_center[1] * center_times_a[1];
    if center_scale <= ELLIPSE_EPS as f64 {
        return None;
    }
    let conic_norm = [
        conic_norm[0] / center_scale,
        conic_norm[1] / center_scale,
        conic_norm[2] / center_scale,
    ];
    let trace = conic_norm[0] + conic_norm[2];
    det_a = conic_norm[0] * conic_norm[2] - conic_norm[1] * conic_norm[1];
    if det_a <= ELLIPSE_EPS as f64
        || conic_norm[0] <= ELLIPSE_EPS as f64
        || conic_norm[2] <= ELLIPSE_EPS as f64
    {
        return None;
    }
    let disc = (0.25 * trace * trace - det_a).max(0.0).sqrt();
    let axis0 = 1.0 / (0.5 * trace + disc).max(ELLIPSE_EPS as f64).sqrt();
    let axis1 = 1.0 / (0.5 * trace - disc).max(ELLIPSE_EPS as f64).sqrt();
    let center_px = [
        bbox_center[0] + ellipse_center[0] as f32 * bbox_half_extent[0],
        bbox_center[1] + ellipse_center[1] as f32 * bbox_half_extent[1],
    ];
    let radius_px =
        (axis0 * bbox_half_extent[0] as f64).max(axis1 * bbox_half_extent[1] as f64) as f32;
    let conic = [
        (conic_norm[0]
            / (bbox_half_extent[0] as f64 * bbox_half_extent[0] as f64).max(ELLIPSE_EPS as f64))
            as f32,
        (conic_norm[1]
            / (bbox_half_extent[0] as f64 * bbox_half_extent[1] as f64).max(ELLIPSE_EPS as f64))
            as f32,
        (conic_norm[2]
            / (bbox_half_extent[1] as f64 * bbox_half_extent[1] as f64).max(ELLIPSE_EPS as f64))
            as f32,
    ];
    if !center_px[0].is_finite()
        || !center_px[1].is_finite()
        || !radius_px.is_finite()
        || !conic[0].is_finite()
        || !conic[1].is_finite()
        || !conic[2].is_finite()
        || radius_px <= 0.0
    {
        return None;
    }
    Some((center_px, radius_px, conic))
}

/// Optional per-splat overrides for numeric backprop (single-splat reproject).
#[derive(Clone, Copy, Default, Debug)]
pub struct SplatProjectOverride {
    pub position: Option<[f32; 3]>,
    pub scale_log: Option<[f32; 3]>,
    pub rotation: Option<[f32; 4]>,
    pub opacity: Option<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn project_splat_index(
    index: usize,
    scene: &GaussianScene,
    camera: &Camera,
    width: u32,
    height: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
    center_radius_depth: &mut [f32],
    ellipse_conic: &mut [f32],
    valid: &mut u32,
    pos_local: &mut [f32],
    inv_scale: &mut [f32],
    override_: SplatProjectOverride,
) {
    let world_pos = override_.position.unwrap_or_else(|| scene.position(index));
    let rotation = override_.rotation.unwrap_or_else(|| scene.rotation(index));
    let sigma = override_.scale_log.unwrap_or_else(|| scene.scale(index));
    let raster_scale = [
        (sigma[0].exp() * radius_scale * GAUSSIAN_SUPPORT_SIGMA_RADIUS).max(1e-6),
        (sigma[1].exp() * radius_scale * GAUSSIAN_SUPPORT_SIGMA_RADIUS).max(1e-6),
        (sigma[2].exp() * radius_scale * GAUSSIAN_SUPPORT_SIGMA_RADIUS).max(1e-6),
    ];
    inv_scale[0] = 1.0 / raster_scale[0];
    inv_scale[1] = 1.0 / raster_scale[1];
    inv_scale[2] = 1.0 / raster_scale[2];
    let delta = [
        camera.position[0] - world_pos[0],
        camera.position[1] - world_pos[1],
        camera.position[2] - world_pos[2],
    ];
    let rotated = quat_rotate(delta, rotation);
    pos_local[0] = rotated[0] * inv_scale[0];
    pos_local[1] = rotated[1] * inv_scale[1];
    pos_local[2] = rotated[2] * inv_scale[2];

    let camera_pos = camera.world_point_to_camera(world_pos);
    let cam_distance = {
        let d = [
            world_pos[0] - camera.position[0],
            world_pos[1] - camera.position[1],
            world_pos[2] - camera.position[2],
        ];
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    };
    let depth_value = camera_pos[2];
    let opacity = override_
        .opacity
        .unwrap_or(scene.opacities[index])
        .clamp(0.0, 1.0);
    if opacity < alpha_cutoff {
        return;
    }
    let support_sigma_radius = (-2.0 * (alpha_cutoff / opacity.max(alpha_cutoff)).ln())
        .max(0.0)
        .sqrt();
    let outline_scale = [
        (sigma[0].exp() * radius_scale * support_sigma_radius).max(1e-6),
        (sigma[1].exp() * radius_scale * support_sigma_radius).max(1e-6),
        (sigma[2].exp() * radius_scale * support_sigma_radius).max(1e-6),
    ];
    let outline_inv_scale = [
        1.0 / outline_scale[0],
        1.0 / outline_scale[1],
        1.0 / outline_scale[2],
    ];
    let fitted = compute_outline_ellipse(
        world_pos,
        outline_inv_scale,
        rotation,
        camera,
        width,
        height,
    );
    let Some((center_px, mut radius_px, conic)) = fitted else {
        return;
    };
    radius_px = (radius_px + ELLIPSE_RADIUS_PAD_PX).max(1.0);
    center_radius_depth[0] = center_px[0];
    center_radius_depth[1] = center_px[1];
    center_radius_depth[2] = radius_px;
    center_radius_depth[3] = cam_distance;
    ellipse_conic[0] = conic[0];
    ellipse_conic[1] = conic[1];
    ellipse_conic[2] = conic[2];
    let visible = depth_value > 1e-4
        && center_px[0] + radius_px >= 0.0
        && center_px[0] - radius_px < width as f32
        && center_px[1] + radius_px >= 0.0
        && center_px[1] - radius_px < height as f32;
    *valid = u32::from(visible);
}

pub fn project_splats(
    scene: &GaussianScene,
    camera: &Camera,
    width: u32,
    height: u32,
    radius_scale: f32,
    alpha_cutoff: f32,
) -> ProjectedSplats {
    let count = scene.count();
    let mut center_radius_depth = vec![0.0f32; count * 4];
    let mut ellipse_conic = vec![0.0f32; count * 3];
    let mut view_dirs = vec![0.0f32; count * 3];
    for i in 0..count {
        let pos = scene.position(i);
        view_dirs[i * 3] = camera.position[0] - pos[0];
        view_dirs[i * 3 + 1] = camera.position[1] - pos[1];
        view_dirs[i * 3 + 2] = camera.position[2] - pos[2];
    }
    let resolved_sh =
        resolve_supported_sh_coeffs(&scene.sh_coeffs, &scene.colors, count, scene.sh_coeff_count);
    let colors = evaluate_sh0_sh1(&resolved_sh, &view_dirs, count);
    let mut color_alpha = vec![0.0f32; count * 4];
    for i in 0..count {
        color_alpha[i * 4] = colors[i * 3];
        color_alpha[i * 4 + 1] = colors[i * 3 + 1];
        color_alpha[i * 4 + 2] = colors[i * 3 + 2];
        color_alpha[i * 4 + 3] = scene.opacities[i];
    }
    let opacity_scale = vec![1.0f32; count];
    let mut valid = vec![0u32; count];
    let mut pos_local = vec![0.0f32; count * 3];
    let mut inv_scale = vec![0.0f32; count * 3];
    let quat = scene.rotations.clone();

    for index in 0..count {
        project_splat_index(
            index,
            scene,
            camera,
            width,
            height,
            radius_scale,
            alpha_cutoff,
            &mut center_radius_depth[index * 4..index * 4 + 4],
            &mut ellipse_conic[index * 3..index * 3 + 3],
            &mut valid[index],
            &mut pos_local[index * 3..index * 3 + 3],
            &mut inv_scale[index * 3..index * 3 + 3],
            SplatProjectOverride::default(),
        );
    }

    ProjectedSplats {
        center_radius_depth,
        ellipse_conic,
        color_alpha,
        opacity_scale,
        valid,
        pos_local,
        inv_scale,
        quat,
    }
}

#[allow(dead_code)]
pub fn default_alpha_cutoff() -> f32 {
    ALPHA_CUTOFF_DEFAULT
}
