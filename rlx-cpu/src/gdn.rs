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
//! Gated-DeltaNet BLAS micro-kernels (Tier C.10).

const MAX_STATE: usize = 128;

/// One recurrent timestep using BLAS (n ≤ 128).
#[inline]
pub fn gdn_step_blas(
    s_mat: &mut [f32],
    q_row: &[f32],
    k_row: &[f32],
    v_row: &[f32],
    g_t: f32,
    beta_t: f32,
    out_row: &mut [f32],
    sk_buf: &mut [f32],
    n: usize,
    scale: f32,
) {
    debug_assert!(n <= MAX_STATE);
    crate::blas::sscal(s_mat, g_t.exp());
    crate::blas::sgemv_at(s_mat, k_row, sk_buf, n, 1.0, 0.0);
    for j in 0..n {
        sk_buf[j] = (v_row[j] - sk_buf[j]) * beta_t;
    }
    crate::blas::sger(s_mat, k_row, sk_buf, n, 1.0);
    crate::blas::sgemv_at(s_mat, q_row, out_row, n, scale, 0.0);
}

pub const GDN_MAX_STATE: usize = MAX_STATE;
