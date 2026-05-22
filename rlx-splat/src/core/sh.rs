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
//! Spherical harmonics helpers aligned with `src/scene/sh_utils.py`.

pub const SH_C0: f32 = 0.28209479177387814;
pub const SH_C1: f32 = 0.4886025119029199;
pub const SH_C2: [f32; 5] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
];
pub const SH_C3: [f32; 7] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
];
pub const SUPPORTED_SH_COEFF_COUNT: usize = 16;

pub fn rgb_to_sh0(colors: [f32; 3]) -> [f32; 3] {
    [
        (colors[0].clamp(0.0, 1.0) - 0.5) / SH_C0,
        (colors[1].clamp(0.0, 1.0) - 0.5) / SH_C0,
        (colors[2].clamp(0.0, 1.0) - 0.5) / SH_C0,
    ]
}

pub fn pad_sh_coeffs(sh_coeffs: &[f32], count: usize, coeff_count: usize) -> Vec<f32> {
    let target = coeff_count.max(1);
    let mut padded = vec![0.0f32; count * target * 3];
    let copy_count = sh_coeffs.len() / (count * 3).max(1);
    let src_coeffs = sh_coeffs.len() / (count * 3);
    let copy_coeffs = src_coeffs.min(target);
    for splat in 0..count {
        for coeff in 0..copy_coeffs {
            for ch in 0..3 {
                padded[splat * target * 3 + coeff * 3 + ch] =
                    sh_coeffs[splat * src_coeffs.max(1) * 3 + coeff * 3 + ch];
            }
        }
    }
    padded
}

pub fn resolve_supported_sh_coeffs(
    sh_coeffs: &[f32],
    colors: &[f32],
    count: usize,
    src_coeff_count: usize,
) -> Vec<f32> {
    let mut resolved = pad_sh_coeffs(sh_coeffs, count, SUPPORTED_SH_COEFF_COUNT);
    if src_coeff_count >= SUPPORTED_SH_COEFF_COUNT {
        return resolved;
    }
    if src_coeff_count == 0 {
        for splat in 0..count {
            let rgb = [
                colors[splat * 3],
                colors[splat * 3 + 1],
                colors[splat * 3 + 2],
            ];
            let sh0 = rgb_to_sh0(rgb);
            resolved[splat * SUPPORTED_SH_COEFF_COUNT * 3] = sh0[0];
            resolved[splat * SUPPORTED_SH_COEFF_COUNT * 3 + 1] = sh0[1];
            resolved[splat * SUPPORTED_SH_COEFF_COUNT * 3 + 2] = sh0[2];
        }
        return resolved;
    }
    let mut all_zero = true;
    for splat in 0..count {
        let base = splat * src_coeff_count * 3;
        if sh_coeffs[base].abs() > 1e-8
            || sh_coeffs[base + 1].abs() > 1e-8
            || sh_coeffs[base + 2].abs() > 1e-8
        {
            all_zero = false;
            break;
        }
    }
    if all_zero {
        for splat in 0..count {
            let rgb = [
                colors[splat * 3],
                colors[splat * 3 + 1],
                colors[splat * 3 + 2],
            ];
            let sh0 = rgb_to_sh0(rgb);
            resolved[splat * SUPPORTED_SH_COEFF_COUNT * 3] = sh0[0];
            resolved[splat * SUPPORTED_SH_COEFF_COUNT * 3 + 1] = sh0[1];
            resolved[splat * SUPPORTED_SH_COEFF_COUNT * 3 + 2] = sh0[2];
        }
    }
    resolved
}

pub fn sh_coeffs_to_display_colors(sh_coeffs: &[f32], count: usize, coeff_count: usize) -> Vec<f32> {
    let mut colors = vec![0.0f32; count * 3];
    for splat in 0..count {
        let base = splat * coeff_count * 3;
        colors[splat * 3] = 0.5 + SH_C0 * sh_coeffs[base];
        colors[splat * 3 + 1] = 0.5 + SH_C0 * sh_coeffs[base + 1];
        colors[splat * 3 + 2] = 0.5 + SH_C0 * sh_coeffs[base + 2];
    }
    colors
}

pub fn evaluate_sh0_sh1(sh_coeffs: &[f32], view_dirs: &[f32], count: usize) -> Vec<f32> {
    let mut colors = vec![0.0f32; count * 3];
    for splat in 0..count {
        let dir = [
            view_dirs[splat * 3],
            view_dirs[splat * 3 + 1],
            view_dirs[splat * 3 + 2],
        ];
        let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        let (x, y, z) = if len > 1e-8 {
            (dir[0] / len, dir[1] / len, dir[2] / len)
        } else {
            (0.0, 0.0, 0.0)
        };
        let base = splat * SUPPORTED_SH_COEFF_COUNT * 3;
        let mut rgb = [
            0.5 + SH_C0 * sh_coeffs[base],
            0.5 + SH_C0 * sh_coeffs[base + 1],
            0.5 + SH_C0 * sh_coeffs[base + 2],
        ];
        if SUPPORTED_SH_COEFF_COUNT > 1 {
            rgb[0] -= SH_C1 * y * sh_coeffs[base + 3];
            rgb[1] -= SH_C1 * y * sh_coeffs[base + 4];
            rgb[2] -= SH_C1 * y * sh_coeffs[base + 5];
            rgb[0] += SH_C1 * z * sh_coeffs[base + 6];
            rgb[1] += SH_C1 * z * sh_coeffs[base + 7];
            rgb[2] += SH_C1 * z * sh_coeffs[base + 8];
            rgb[0] -= SH_C1 * x * sh_coeffs[base + 9];
            rgb[1] -= SH_C1 * x * sh_coeffs[base + 10];
            rgb[2] -= SH_C1 * x * sh_coeffs[base + 11];
        }
        colors[splat * 3] = rgb[0];
        colors[splat * 3 + 1] = rgb[1];
        colors[splat * 3 + 2] = rgb[2];
    }
    colors
}
