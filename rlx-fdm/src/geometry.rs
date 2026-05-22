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

//! Geometry helpers for tributary loads (jax_fdm `geometry`).

use crate::network::Vec3;

const WORLD_X: [f64; 3] = [1.0, 0.0, 0.0];
const WORLD_Y: [f64; 3] = [0.0, 1.0, 0.0];
const WORLD_Z: [f64; 3] = [0.0, 0.0, 1.0];

pub fn length(v: Vec3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

pub fn normalize(v: Vec3) -> Vec3 {
    let mut u = v;
    if u.iter().all(|&x| x == 0.0 || x.is_nan()) {
        return [0.0; 3];
    }
    let l = length(u).max(1e-14);
    u[0] /= l;
    u[1] /= l;
    u[2] /= l;
    u
}

pub fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Angle in radians between two vectors (jax_fdm `angle_vectors`).
pub fn angle_vectors(a: Vec3, b: Vec3) -> f64 {
    let na = normalize(a);
    let nb = normalize(b);
    dot(na, nb).clamp(-1.0, 1.0).acos()
}

/// Unnormalized polygon normal (jax_fdm `normal_polygon`).
pub fn normal_polygon(points: &[Vec3]) -> Vec3 {
    if points.len() < 3 {
        return [0.0; 3];
    }
    let mut n = [0.0; 3];
    for i in 0..points.len() {
        let j = (i + 1) % points.len();
        let a = points[i];
        let b = points[j];
        let c = cross(a, b);
        n[0] += c[0];
        n[1] += c[1];
        n[2] += c[2];
    }
    n
}

pub fn area_triangle(triangle: [Vec3; 3]) -> f64 {
    0.5 * length(normal_triangle(triangle))
}

fn normal_triangle(triangle: [Vec3; 3]) -> Vec3 {
    let line_a = [
        triangle[0][0] - triangle[1][0],
        triangle[0][1] - triangle[1][1],
        triangle[0][2] - triangle[1][2],
    ];
    let line_b = [
        triangle[2][0] - triangle[1][0],
        triangle[2][1] - triangle[1][1],
        triangle[2][2] - triangle[1][2],
    ];
    cross(line_a, line_b)
}

/// 3×3 LCS rows `[u; v; w]` (jax_fdm `polygon_lcs`).
pub fn polygon_lcs(polygon: &[Vec3]) -> [[f64; 3]; 3] {
    let w = normalize(normal_polygon(polygon));
    let threshold = dot(w, WORLD_X).abs() > 0.999;
    let vperp = if threshold { WORLD_Y } else { WORLD_X };
    let mut v = cross(w, vperp);
    v = normalize(v);
    let mut u = cross(w, v);
    if dot(u, vperp) < 0.0 {
        u = [-u[0], -u[1], -u[2]];
    }
    u = normalize(u);
    [u, v, w]
}

/// Triangle area gradient w.r.t. vertices `a`, `b`, `c` (area = ½‖(b−a)×(c−a)‖).
pub fn grad_area_triangle(a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3, Vec3) {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let n = cross(u, v);
    let len = length(n).max(1e-14);
    let nh = [n[0] / len, n[1] / len, n[2] / len];
    let ga = [
        0.5 * (v[1] * nh[2] - v[2] * nh[1]),
        0.5 * (v[2] * nh[0] - v[0] * nh[2]),
        0.5 * (v[0] * nh[1] - v[1] * nh[0]),
    ];
    let gb = [
        0.5 * (nh[1] * u[2] - nh[2] * u[1]),
        0.5 * (nh[2] * u[0] - nh[0] * u[2]),
        0.5 * (nh[0] * u[1] - nh[1] * u[0]),
    ];
    let gc = [
        0.5 * (u[1] * nh[2] - u[2] * nh[1]),
        0.5 * (u[2] * nh[0] - u[0] * nh[2]),
        0.5 * (u[0] * nh[1] - u[1] * nh[0]),
    ];
    (ga, gb, gc)
}

/// Face planarity metric (jax_fdm `polygon_planarity`): Σ |n̂·ê| over unit edge directions.
pub fn polygon_planarity(polygon: &[Vec3]) -> f64 {
    if polygon.len() < 3 {
        return 0.0;
    }
    let n = normalize(normal_polygon(polygon));
    let mut s = 0.0;
    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        let e = [
            polygon[j][0] - polygon[i][0],
            polygon[j][1] - polygon[i][1],
            polygon[j][2] - polygon[i][2],
        ];
        let el = length(e);
        if el < 1e-14 {
            continue;
        }
        let en = [e[0] / el, e[1] / el, e[2] / el];
        s += dot(n, en).abs();
    }
    s
}

/// Corner orthogonality metric (jax_fdm `FaceRectangularGoal`): Σ (ê₁·ê₂)² at each polygon vertex.
pub fn polygon_rectangular_deviation(polygon: &[Vec3]) -> f64 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }
    let mut s = 0.0;
    for i in 0..n {
        let im = (i + n - 1) % n;
        let ip = (i + 1) % n;
        let mut e1 = [
            polygon[i][0] - polygon[im][0],
            polygon[i][1] - polygon[im][1],
            polygon[i][2] - polygon[im][2],
        ];
        let mut e2 = [
            polygon[ip][0] - polygon[i][0],
            polygon[ip][1] - polygon[i][1],
            polygon[ip][2] - polygon[i][2],
        ];
        let l1 = length(e1).max(1e-14);
        let l2 = length(e2).max(1e-14);
        e1 = [e1[0] / l1, e1[1] / l1, e1[2] / l1];
        e2 = [e2[0] / l2, e2[1] / l2, e2[2] / l2];
        let d = dot(e1, e2);
        s += d * d;
    }
    s
}

/// `∂(rectangular deviation)/∂x` per polygon vertex (FD).
pub fn grad_polygon_rectangular_wrt_nodes(
    polygon: &[Vec3],
    node_indices: &[usize],
) -> Vec<(usize, Vec3)> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let eps = 1e-8;
    let base = polygon_rectangular_deviation(polygon);
    let mut out = Vec::new();
    for (vi, &node) in node_indices.iter().enumerate() {
        let mut g = [0.0; 3];
        for d in 0..3 {
            let mut pert = polygon.to_vec();
            pert[vi][d] += eps;
            g[d] = (polygon_rectangular_deviation(&pert) - base) / eps;
        }
        out.push((node, g));
    }
    out
}

/// `∂planarity/∂x` for face vertices, packed as contributions per global node index.
pub fn grad_polygon_planarity_wrt_nodes(
    polygon: &[Vec3],
    node_indices: &[usize],
) -> Vec<(usize, Vec3)> {
    if polygon.len() < 3 {
        return Vec::new();
    }
    let eps = 1e-8;
    let base = polygon_planarity(polygon);
    let mut out = Vec::new();
    for (vi, &node) in node_indices.iter().enumerate() {
        let mut g = [0.0; 3];
        for d in 0..3 {
            let mut pert = polygon.to_vec();
            pert[vi][d] += eps;
            g[d] = (polygon_planarity(&pert) - base) / eps;
        }
        out.push((node, g));
    }
    out
}

/// Transform face load from local to global (`face_load @ lcs`).
pub fn face_load_global(polygon: &[Vec3], face_load: Vec3) -> Vec3 {
    let lcs = polygon_lcs(polygon);
    mat_vec3_mul_rows(lcs, face_load)
}

fn mat_vec3_mul_rows(rows: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    [
        v[0] * rows[0][0] + v[1] * rows[1][0] + v[2] * rows[2][0],
        v[0] * rows[0][1] + v[1] * rows[1][1] + v[2] * rows[2][1],
        v[0] * rows[0][2] + v[1] * rows[1][2] + v[2] * rows[2][2],
    ]
}

/// Per polygon vertex `vi`: `j[c][d] = ∂(face_load_global)_c / ∂(vertex vi)_d`.
pub fn jacobian_face_load_global_wrt_polygon(
    polygon: &[Vec3],
    face_load: Vec3,
) -> Vec<[[f64; 3]; 3]> {
    if polygon.len() < 3 {
        return vec![[[0.0; 3]; 3]; polygon.len()];
    }
    let eps = 1e-8;
    let mut jac = vec![[[0.0; 3]; 3]; polygon.len()];
    for vi in 0..polygon.len() {
        for d in 0..3 {
            let mut plus = polygon.to_vec();
            let mut minus = polygon.to_vec();
            plus[vi][d] += eps;
            minus[vi][d] -= eps;
            let fp = face_load_global(&plus, face_load);
            let fm = face_load_global(&minus, face_load);
            for c in 0..3 {
                jac[vi][c][d] = (fp[c] - fm[c]) / (2.0 * eps);
            }
        }
    }
    jac
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn face_load_global_jacobian_matches_fd() {
        let poly = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.2],
        ];
        let load = [0.0, 0.0, -1.0];
        let j = jacobian_face_load_global_wrt_polygon(&poly, load);
        let eps = 1e-7;
        for vi in 0..poly.len() {
            for d in 0..3 {
                let mut p = poly.to_vec();
                p[vi][d] += eps;
                let fp = face_load_global(&p, load);
                let base = face_load_global(&poly, load);
                for c in 0..3 {
                    let fd = (fp[c] - base[c]) / eps;
                    assert!(
                        (j[vi][c][d] - fd).abs() < 1e-5,
                        "vi={vi} c={c} d={d}: jac={} fd={}",
                        j[vi][c][d],
                        fd
                    );
                }
            }
        }
    }
}
