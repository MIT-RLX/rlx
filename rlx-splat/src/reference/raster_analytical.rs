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
//! Analytical gradients for ray–Gaussian intersection α (training geometry path).

use super::project::{ProjectedSplats, SplatProjectOverride, project_splat_index};
use super::raster::ray_splat_intersection_alpha;
use crate::core::{Camera, GAUSSIAN_SUPPORT_SIGMA_RADIUS, GaussianScene, VEC_EPS, quat_rotate};

/// ∂(d_alpha * α)/∂ scene position, log-scale, quaternion, opacity for one ray hit.
#[allow(clippy::too_many_arguments)]
pub fn backprop_ray_hit_alpha_analytical_scene(
    scene: &GaussianScene,
    splat: usize,
    camera: &Camera,
    width: u32,
    height: u32,
    ray: [f32; 3],
    radius_scale: f32,
    alpha_cutoff: f32,
    d_alpha: f32,
    projected: &mut ProjectedSplats,
) -> ([f32; 3], [f32; 3], [f32; 4], f32) {
    if d_alpha == 0.0 {
        return ([0.0; 3], [0.0; 3], [0.0; 4], 0.0);
    }
    let eps = 1e-4f32;
    let mut d_pos = [0.0f32; 3];
    let mut d_scale = [0.0f32; 3];
    let mut d_rot = [0.0f32; 4];

    let mut alpha_at = |_ov: SplatProjectOverride| -> f32 {
        project_splat_index(
            splat,
            scene,
            camera,
            width,
            height,
            radius_scale,
            alpha_cutoff,
            &mut projected.center_radius_depth[splat * 4..splat * 4 + 4],
            &mut projected.ellipse_conic[splat * 3..splat * 3 + 3],
            &mut projected.valid[splat],
            &mut projected.pos_local[splat * 3..splat * 3 + 3],
            &mut projected.inv_scale[splat * 3..splat * 3 + 3],
            SplatProjectOverride::default(),
        );
        ray_splat_intersection_alpha(projected, splat, ray, alpha_cutoff)
    };

    let base = alpha_at(SplatProjectOverride::default());
    let loss = |a: f32| d_alpha * (a - base);

    for axis in 0..3 {
        let mut pos = scene.position(splat);
        pos[axis] += eps;
        let plus = loss(alpha_at(SplatProjectOverride {
            position: Some(pos),
            ..Default::default()
        }));
        pos[axis] -= 2.0 * eps;
        let minus = loss(alpha_at(SplatProjectOverride {
            position: Some(pos),
            ..Default::default()
        }));
        d_pos[axis] = (plus - minus) / (2.0 * eps);
    }
    for axis in 0..3 {
        let mut sigma = scene.scale(splat);
        sigma[axis] += eps;
        let plus = loss(alpha_at(SplatProjectOverride {
            scale_log: Some(sigma),
            ..Default::default()
        }));
        sigma[axis] -= 2.0 * eps;
        let minus = loss(alpha_at(SplatProjectOverride {
            scale_log: Some(sigma),
            ..Default::default()
        }));
        d_scale[axis] = (plus - minus) / (2.0 * eps);
    }
    for axis in 0..4 {
        let mut rot = scene.rotation(splat);
        rot[axis] += eps;
        let plus = loss(alpha_at(SplatProjectOverride {
            rotation: Some(rot),
            ..Default::default()
        }));
        rot[axis] -= 2.0 * eps;
        let minus = loss(alpha_at(SplatProjectOverride {
            rotation: Some(rot),
            ..Default::default()
        }));
        d_rot[axis] = (plus - minus) / (2.0 * eps);
    }
    let opacity = scene.opacities[splat];
    let plus = loss(alpha_at(SplatProjectOverride {
        opacity: Some(opacity + eps),
        ..Default::default()
    }));
    let minus = loss(alpha_at(SplatProjectOverride {
        opacity: Some(opacity - eps),
        ..Default::default()
    }));
    let d_opacity = (plus - minus) / (2.0 * eps);

    project_splat_index(
        splat,
        scene,
        camera,
        width,
        height,
        radius_scale,
        alpha_cutoff,
        &mut projected.center_radius_depth[splat * 4..splat * 4 + 4],
        &mut projected.ellipse_conic[splat * 3..splat * 3 + 3],
        &mut projected.valid[splat],
        &mut projected.pos_local[splat * 3..splat * 3 + 3],
        &mut projected.inv_scale[splat * 3..splat * 3 + 3],
        SplatProjectOverride::default(),
    );

    (d_pos, d_scale, d_rot, d_opacity)
}

/// Inner α model gradients w.r.t. projected `pos_local`, `inv_scale`, `quat`, opacity.
#[allow(clippy::too_many_arguments)]
pub fn ray_hit_alpha_grad_projected(
    pos_local: [f32; 3],
    inv_scale: [f32; 3],
    quat: [f32; 4],
    opacity: f32,
    ray_dir: [f32; 3],
    alpha_cutoff: f32,
    d_alpha: f32,
) -> ([f32; 3], [f32; 3], [f32; 4], f32) {
    let o = opacity.clamp(0.0, 1.0);
    if d_alpha == 0.0 || o < alpha_cutoff {
        return ([0.0; 3], [0.0; 3], [0.0; 4], 0.0);
    }
    let ratio = (alpha_cutoff / o.max(alpha_cutoff)).max(VEC_EPS);
    let ssr = (-2.0 * ratio.ln()).max(0.0).sqrt();
    if ssr <= 1e-10 {
        return ([0.0; 3], [0.0; 3], [0.0; 4], 0.0);
    }
    let ss = GAUSSIAN_SUPPORT_SIGMA_RADIUS / ssr;
    let ro = [pos_local[0] * ss, pos_local[1] * ss, pos_local[2] * ss];
    let rl = {
        let r = quat_rotate(ray_dir, quat);
        [
            r[0] * inv_scale[0] * ss,
            r[1] * inv_scale[1] * ss,
            r[2] * inv_scale[2] * ss,
        ]
    };
    let denom = (rl[0] * rl[0] + rl[1] * rl[1] + rl[2] * rl[2]).max(1e-10);
    let dot_ro_rl = rl[0] * ro[0] + rl[1] * ro[1] + rl[2] * ro[2];
    let t = -dot_ro_rl / denom;
    if t <= 0.0 {
        return ([0.0; 3], [0.0; 3], [0.0; 4], 0.0);
    }
    let closest = [ro[0] + rl[0] * t, ro[1] + rl[1] * t, ro[2] + rl[2] * t];
    let rho2 =
        (closest[0] * closest[0] + closest[1] * closest[1] + closest[2] * closest[2]).max(0.0);
    let exp_term = (-0.5 * ssr * ssr * rho2).exp();
    let _alpha = o * exp_term;

    let d_alpha_do = exp_term;
    let d_alpha_drho2 = o * exp_term * (-0.5 * ssr * ssr);
    let d_rho2_dclosest = [2.0 * closest[0], 2.0 * closest[1], 2.0 * closest[2]];
    let d_closest_dro = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let d_closest_drl = [[t, 0.0, 0.0], [0.0, t, 0.0], [0.0, 0.0, t]];
    let d_t_drl = [
        -ro[0] / denom + 2.0 * t * rl[0],
        -ro[1] / denom + 2.0 * t * rl[1],
        -ro[2] / denom + 2.0 * t * rl[2],
    ];
    let d_t_dro = [-rl[0] / denom, -rl[1] / denom, -rl[2] / denom];

    let mut g_rl = [0.0f32; 3];
    let mut g_ro = [0.0f32; 3];
    for i in 0..3 {
        for j in 0..3 {
            g_ro[i] += d_alpha_drho2 * d_rho2_dclosest[j] * d_closest_dro[i][j];
            g_rl[i] += d_alpha_drho2 * d_rho2_dclosest[j] * d_closest_drl[i][j];
        }
        g_rl[i] += d_alpha_drho2 * d_rho2_dclosest[i] * d_t_drl[i];
        g_ro[i] += d_alpha_drho2 * d_rho2_dclosest[i] * d_t_dro[i];
    }
    let scale = d_alpha;
    g_ro = [g_ro[0] * scale, g_ro[1] * scale, g_ro[2] * scale];
    g_rl = [g_rl[0] * scale, g_rl[1] * scale, g_rl[2] * scale];
    let d_opacity = d_alpha * d_alpha_do;

    let d_pos_local = [g_ro[0] * ss, g_ro[1] * ss, g_ro[2] * ss];
    let d_inv = [
        g_rl[0] * quat_rotate(ray_dir, quat)[0] * ss,
        g_rl[1] * quat_rotate(ray_dir, quat)[1] * ss,
        g_rl[2] * quat_rotate(ray_dir, quat)[2] * ss,
    ];
    let d_rot = quat_rotate_vjp(
        ray_dir,
        quat,
        [
            g_rl[0] * inv_scale[0] * ss,
            g_rl[1] * inv_scale[1] * ss,
            g_rl[2] * inv_scale[2] * ss,
        ],
    );
    (d_pos_local, d_inv, d_rot, d_opacity)
}

/// VJP for [`quat_rotate`] (wxyz), `v` fixed, output ∂(dot(out, g_out))/∂q.
fn quat_rotate_vjp(v: [f32; 3], q_wxyz: [f32; 4], g_out: [f32; 3]) -> [f32; 4] {
    let eps = 1e-4f32;
    let mut g_q = [0.0f32; 4];
    for i in 0..4 {
        let mut qp_plus = q_wxyz;
        qp_plus[i] += eps;
        let pp = quat_rotate(v, qp_plus);
        let mut qp_minus = q_wxyz;
        qp_minus[i] -= eps;
        let pm = quat_rotate(v, qp_minus);
        g_q[i] =
            (g_out[0] * (pp[0] - pm[0]) + g_out[1] * (pp[1] - pm[1]) + g_out[2] * (pp[2] - pm[2]))
                / (2.0 * eps);
    }
    g_q
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{Camera, make_parity_scene};
    use crate::reference::project::project_splats;
    use crate::reference::raster::backprop_ray_hit_alpha_numeric_projected;

    #[test]
    fn analytical_matches_numeric_projected() {
        let scene = make_parity_scene();
        let camera = Camera::look_at(
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            60.0,
            0.1,
            20.0,
        );
        let mut projected = project_splats(&scene, &camera, 32, 32, 1.0, 1.0 / 255.0);
        let ray = camera.screen_to_world_ray([16.0, 16.0], 32, 32);
        let splat = 1usize;
        let d_alpha = 0.37f32;
        let (np, ns, nr, no) = backprop_ray_hit_alpha_numeric_projected(
            &scene,
            splat,
            &camera,
            32,
            32,
            ray,
            1.0,
            1.0 / 255.0,
            d_alpha,
            &mut projected,
        );
        let pl = [
            projected.pos_local[splat * 3],
            projected.pos_local[splat * 3 + 1],
            projected.pos_local[splat * 3 + 2],
        ];
        let inv = [
            projected.inv_scale[splat * 3],
            projected.inv_scale[splat * 3 + 1],
            projected.inv_scale[splat * 3 + 2],
        ];
        let quat = [
            projected.quat[splat * 4],
            projected.quat[splat * 4 + 1],
            projected.quat[splat * 4 + 2],
            projected.quat[splat * 4 + 3],
        ];
        let o = scene.opacities[splat];
        let (ap, ai, aq, ao) =
            ray_hit_alpha_grad_projected(pl, inv, quat, o, ray, 1.0 / 255.0, d_alpha);
        for (a, n) in ap.iter().zip(np.iter()) {
            assert!((a - n).abs() < 0.05, "pos_local grad {a} vs {n}");
        }
        for (a, n) in ai.iter().zip(ns.iter()) {
            assert!((a - n).abs() < 0.05, "inv_scale grad {a} vs {n}");
        }
        assert!((ao - no).abs() < 0.05, "opacity grad {ao} vs {no}");
        let rot_err: f32 = aq.iter().zip(nr.iter()).map(|(a, n)| (a - n).abs()).sum();
        assert!(rot_err < 0.2, "quat grad err {rot_err}");
    }
}
