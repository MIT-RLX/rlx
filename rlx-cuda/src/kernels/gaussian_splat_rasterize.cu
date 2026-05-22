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

struct SplatRasterParams {
    unsigned int width;
    unsigned int height;
    unsigned int tile_size;
    unsigned int tile_width;
    float alpha_cutoff;
    float transmittance_threshold;
    float bg_r;
    float bg_g;
    float bg_b;
};

__device__ inline float3 quat_rotate(float3 v, float4 q_wxyz) {
    float3 qv = make_float3(q_wxyz.y, q_wxyz.z, q_wxyz.w);
    float w = q_wxyz.x;
    float3 t1 = make_float3(
        v.y * qv.z - v.z * qv.y,
        v.z * qv.x - v.x * qv.z,
        v.x * qv.y - v.y * qv.x);
    float3 mid = make_float3(t1.x + w * v.x, t1.y + w * v.y, t1.z + w * v.z);
    float3 t2 = make_float3(
        mid.y * qv.z - mid.z * qv.y,
        mid.z * qv.x - mid.x * qv.z,
        mid.x * qv.y - mid.y * qv.x);
    return make_float3(v.x + 2.f * t2.x, v.y + 2.f * t2.y, v.z + 2.f * t2.z);
}

__device__ inline float ray_splat_intersection_alpha(
    const float* color_alpha,
    const float* pos_local,
    const float* inv_scale,
    const float* quat,
    unsigned int splat_id,
    float3 ray_direction,
    float alpha_cutoff
) {
    const float kSupport = 3.f;
    unsigned int ca = splat_id * 4u;
    float opacity = fminf(fmaxf(color_alpha[ca + 3], 0.f), 1.f);
    if (opacity < alpha_cutoff) return 0.f;
    float support_sigma_radius = sqrtf(fmaxf(0.f, -2.f * logf(alpha_cutoff / fmaxf(opacity, alpha_cutoff))));
    if (support_sigma_radius <= 1e-10f) return 0.f;
    float support_scale = kSupport / support_sigma_radius;
    unsigned int pl = splat_id * 3u;
    float3 ro_local = make_float3(
        pos_local[pl] * support_scale,
        pos_local[pl + 1] * support_scale,
        pos_local[pl + 2] * support_scale);
    unsigned int qb = splat_id * 4u;
    float4 q_wxyz = make_float4(quat[qb], quat[qb + 1], quat[qb + 2], quat[qb + 3]);
    unsigned int is = splat_id * 3u;
    float3 inv = make_float3(inv_scale[is], inv_scale[is + 1], inv_scale[is + 2]);
    float3 rotated = quat_rotate(ray_direction, q_wxyz);
    float3 ray_local = make_float3(
        rotated.x * inv.x * support_scale,
        rotated.y * inv.y * support_scale,
        rotated.z * inv.z * support_scale);
    float denom = ray_local.x * ray_local.x + ray_local.y * ray_local.y + ray_local.z * ray_local.z;
    if (denom <= 1e-10f) return 0.f;
    float t_closest = -(ray_local.x * ro_local.x + ray_local.y * ro_local.y + ray_local.z * ro_local.z) / denom;
    if (t_closest <= 0.f) return 0.f;
    float3 closest = make_float3(
        ro_local.x + ray_local.x * t_closest,
        ro_local.y + ray_local.y * t_closest,
        ro_local.z + ray_local.z * t_closest);
    float rho2 = fmaxf(0.f, closest.x * closest.x + closest.y * closest.y + closest.z * closest.z);
    return opacity * expf(-0.5f * support_sigma_radius * support_sigma_radius * rho2);
}

extern "C" __global__ void gaussian_splat_rasterize(
    float* arena,
    unsigned int dst_off,
    const float* color_alpha,
    const unsigned int* valid,
    const float* pos_local,
    const float* inv_scale,
    const float* quat,
    const unsigned int* sorted_values,
    const unsigned int* tile_ranges,
    const float* rays,
    SplatRasterParams params
) {
    const float kGamma = 2.2f;
    float* dst = arena + dst_off;
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= params.width || py >= params.height) return;
    unsigned int out_base = (py * params.width + px) * 4u;
    unsigned int tile_y = py / params.tile_size;
    unsigned int tile = tile_y * params.tile_width + px / params.tile_size;
    unsigned int range_base = tile * 2u;
    unsigned int start = tile_ranges[range_base];
    unsigned int end = tile_ranges[range_base + 1u];
    if (start == 0xFFFFFFFFu || end <= start) {
        dst[out_base] = params.bg_r;
        dst[out_base + 1] = params.bg_g;
        dst[out_base + 2] = params.bg_b;
        dst[out_base + 3] = 1.f;
        return;
    }
    unsigned int ray_base = (py * params.width + px) * 3u;
    float3 ray = make_float3(rays[ray_base], rays[ray_base + 1], rays[ray_base + 2]);
    float accum_x = 0.f, accum_y = 0.f, accum_z = 0.f;
    float trans = 1.f;
    for (unsigned int i = start; i < end; ++i) {
        unsigned int splat_id = sorted_values[i];
        if (valid[splat_id] == 0u) continue;
        float alpha = ray_splat_intersection_alpha(
            color_alpha, pos_local, inv_scale, quat, splat_id, ray, params.alpha_cutoff);
        if (alpha < params.alpha_cutoff) continue;
        unsigned int rgb_base = splat_id * 4u;
        accum_x += trans * alpha * color_alpha[rgb_base];
        accum_y += trans * alpha * color_alpha[rgb_base + 1];
        accum_z += trans * alpha * color_alpha[rgb_base + 2];
        trans *= 1.f - alpha;
        if (trans < params.transmittance_threshold) break;
    }
    float cx = fmaxf(0.f, accum_x + trans * params.bg_r);
    float cy = fmaxf(0.f, accum_y + trans * params.bg_g);
    float cz = fmaxf(0.f, accum_z + trans * params.bg_b);
    dst[out_base] = powf(cx, kGamma);
    dst[out_base + 1] = powf(cy, kGamma);
    dst[out_base + 2] = powf(cz, kGamma);
    dst[out_base + 3] = 1.f - trans;
}
