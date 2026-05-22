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
//! Small vector helpers mirroring `src/utility/math.py`.

use super::constants::VEC_EPS;

#[inline]
pub fn normalize3(v: [f32; 3], eps: f32) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len <= eps {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[inline]
pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
pub fn mat3_mul_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
pub fn mat3_transpose_mul_vec3(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
        m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
        m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
pub fn quat_rotate(v: [f32; 3], q_wxyz: [f32; 4]) -> [f32; 3] {
    let qv = [q_wxyz[1], q_wxyz[2], q_wxyz[3]];
    let w = q_wxyz[0];
    let t1 = cross(v, qv);
    let mid = [t1[0] + w * v[0], t1[1] + w * v[1], t1[2] + w * v[2]];
    let t2 = cross(mid, qv);
    [v[0] + 2.0 * t2[0], v[1] + 2.0 * t2[1], v[2] + 2.0 * t2[2]]
}

#[inline]
pub fn quat_conj(q_wxyz: [f32; 4]) -> [f32; 4] {
    [q_wxyz[0], -q_wxyz[1], -q_wxyz[2], -q_wxyz[3]]
}

#[inline]
pub fn quat_normalize(q_wxyz: [f32; 4]) -> [f32; 4] {
    let len = (q_wxyz[0] * q_wxyz[0]
        + q_wxyz[1] * q_wxyz[1]
        + q_wxyz[2] * q_wxyz[2]
        + q_wxyz[3] * q_wxyz[3])
        .sqrt()
        .max(VEC_EPS);
    [
        q_wxyz[0] / len,
        q_wxyz[1] / len,
        q_wxyz[2] / len,
        q_wxyz[3] / len,
    ]
}

#[inline]
pub fn rotation_matrix_from_quaternion_wxyz(q_wxyz: [f32; 4]) -> [[f32; 3]; 3] {
    let q = quat_normalize(q_wxyz);
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

pub fn quaternion_from_rotation_matrix(m: [[f32; 3]; 3]) -> [f32; 4] {
    let trace = m[0][0] + m[1][1] + m[2][2];
    let (w, x, y, z) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        )
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
        (
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        )
    } else if m[1][1] > m[2][2] {
        let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
        (
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        )
    } else {
        let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
        (
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        )
    };
    quat_normalize([w, x, y, z])
}

#[inline]
pub fn log_sigma(sigma: f32) -> f32 {
    sigma.ln()
}

#[inline]
pub fn exp_log_scale(log_scale: [f32; 3]) -> [f32; 3] {
    [log_scale[0].exp(), log_scale[1].exp(), log_scale[2].exp()]
}
