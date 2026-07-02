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

//! Equilibrium results (jax_fdm `equilibrium.states`).

/// Static equilibrium state after one FDM solve.
#[derive(Clone, Debug, PartialEq)]
pub struct EquilibriumState {
    /// All node coordinates `[n, 3]` flattened.
    pub xyz: Vec<f64>,
    pub lengths: Vec<f64>,
    pub forces: Vec<f64>,
    /// Unbalanced force per node `[n, 3]` (should be ~0 at free nodes).
    pub residuals: Vec<f64>,
    pub loads: Vec<f64>,
    /// Edge direction vectors (unnormalized).
    pub vectors: Vec<f64>,
}

impl EquilibriumState {
    pub fn num_nodes(&self) -> usize {
        self.xyz.len() / 3
    }

    pub fn max_free_residual_norm(&self, is_support: &[bool]) -> f64 {
        let mut max_r: f64 = 0.0;
        for i in 0..self.num_nodes() {
            if is_support[i] {
                continue;
            }
            let rx = self.residuals[3 * i];
            let ry = self.residuals[3 * i + 1];
            let rz = self.residuals[3 * i + 2];
            max_r = max_r.max((rx * rx + ry * ry + rz * rz).sqrt());
        }
        max_r
    }
}
