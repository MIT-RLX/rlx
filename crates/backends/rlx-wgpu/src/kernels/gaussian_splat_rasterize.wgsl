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
// RLX native Gaussian splat tile raster — matches rlx-metal/src/splat.msl.

const GAUSSIAN_SUPPORT_SIGMA_RADIUS: f32 = 3.0;
const OUTPUT_GAMMA: f32 = 2.2;

struct SplatRasterParams {
    width: u32,
    height: u32,
    tile_size: u32,
    tile_width: u32,
    alpha_cutoff: f32,
    transmittance_threshold: f32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    dst_base: u32,
}

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<storage, read> color_alpha: array<f32>;
@group(0) @binding(2) var<storage, read> valid: array<u32>;
@group(0) @binding(3) var<storage, read> pos_local: array<f32>;
@group(0) @binding(4) var<storage, read> inv_scale: array<f32>;
@group(0) @binding(5) var<storage, read> quat: array<f32>;
@group(0) @binding(6) var<storage, read> sorted_values: array<u32>;
@group(0) @binding(7) var<storage, read> tile_ranges: array<u32>;
@group(0) @binding(8) var<storage, read> rays: array<f32>;
@group(0) @binding(9) var<uniform> params: SplatRasterParams;

fn quat_rotate(v: vec3f, q_wxyz: vec4f) -> vec3f {
    let qv = q_wxyz.yzw;
    let w = q_wxyz.x;
    let t1 = cross(v, qv);
    let mid = t1 + w * v;
    let t2 = cross(mid, qv);
    return v + 2.0 * t2;
}

fn ray_splat_intersection_alpha(
    splat_id: u32,
    ray_direction: vec3f,
    alpha_cutoff: f32,
) -> f32 {
    let ca = splat_id * 4u;
    let opacity = clamp(color_alpha[ca + 3u], 0.0, 1.0);
    if (opacity < alpha_cutoff) {
        return 0.0;
    }
    let support_sigma_radius = sqrt(max(0.0, -2.0 * log(alpha_cutoff / max(opacity, alpha_cutoff))));
    if (support_sigma_radius <= 1e-10) {
        return 0.0;
    }
    let support_scale = GAUSSIAN_SUPPORT_SIGMA_RADIUS / support_sigma_radius;
    let pl = splat_id * 3u;
    let ro_local = vec3f(pos_local[pl], pos_local[pl + 1u], pos_local[pl + 2u]) * support_scale;
    let qb = splat_id * 4u;
    let q_wxyz = vec4f(quat[qb], quat[qb + 1u], quat[qb + 2u], quat[qb + 3u]);
    let is = splat_id * 3u;
    let inv = vec3f(inv_scale[is], inv_scale[is + 1u], inv_scale[is + 2u]);
    let rotated = quat_rotate(ray_direction, q_wxyz);
    let ray_local = vec3f(rotated.x * inv.x, rotated.y * inv.y, rotated.z * inv.z) * support_scale;
    let denom = dot(ray_local, ray_local);
    if (denom <= 1e-10) {
        return 0.0;
    }
    let t_closest = -dot(ray_local, ro_local) / denom;
    if (t_closest <= 0.0) {
        return 0.0;
    }
    let closest = ro_local + ray_local * t_closest;
    let rho2 = max(0.0, dot(closest, closest));
    return opacity * exp(-0.5 * support_sigma_radius * support_sigma_radius * rho2);
}

@compute @workgroup_size(8, 8, 1)
fn gaussian_splat_rasterize(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.width || py >= params.height) {
        return;
    }
    let out_base = params.dst_base + (py * params.width + px) * 4u;
    let tile_y = py / params.tile_size;
    let tile = tile_y * params.tile_width + px / params.tile_size;
    let range_base = tile * 2u;
    let start = tile_ranges[range_base];
    let end = tile_ranges[range_base + 1u];
    if (start == 0xFFFFFFFFu || end <= start) {
        dst[out_base] = params.bg_r;
        dst[out_base + 1u] = params.bg_g;
        dst[out_base + 2u] = params.bg_b;
        dst[out_base + 3u] = 1.0;
        return;
    }
    let ray_base = (py * params.width + px) * 3u;
    let ray = vec3f(rays[ray_base], rays[ray_base + 1u], rays[ray_base + 2u]);
    var accum = vec3f(0.0);
    var trans = 1.0;
    for (var i = start; i < end; i++) {
        let splat_id = sorted_values[i];
        if (valid[splat_id] == 0u) {
            continue;
        }
        let alpha = ray_splat_intersection_alpha(splat_id, ray, params.alpha_cutoff);
        if (alpha < params.alpha_cutoff) {
            continue;
        }
        let rgb_base = splat_id * 4u;
        accum += trans * alpha * vec3f(
            color_alpha[rgb_base],
            color_alpha[rgb_base + 1u],
            color_alpha[rgb_base + 2u],
        );
        trans *= 1.0 - alpha;
        if (trans < params.transmittance_threshold) {
            break;
        }
    }
    let bg = vec3f(params.bg_r, params.bg_g, params.bg_b);
    let composed = max(vec3f(0.0), accum + trans * bg);
    dst[out_base] = pow(composed.x, OUTPUT_GAMMA);
    dst[out_base + 1u] = pow(composed.y, OUTPUT_GAMMA);
    dst[out_base + 2u] = pow(composed.z, OUTPUT_GAMMA);
    dst[out_base + 3u] = 1.0 - trans;
}
