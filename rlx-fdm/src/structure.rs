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

//! Incidence / connectivity (jax_fdm `equilibrium.structures` + COMPAS `connectivity_matrix`).

use crate::network::Network;

/// Static topology derived from a [`Network`].
#[derive(Clone, Debug)]
pub struct Structure {
    pub num_nodes: usize,
    pub num_edges: usize,
    /// Full incidence `C` with shape `[num_edges, num_nodes]`: +1 at tail, −1 at head.
    pub connectivity: Vec<f64>,
    /// Indices of free (unsupported) nodes in ascending key order.
    pub indices_free: Vec<usize>,
    /// Indices of fixed (supported) nodes.
    pub indices_fixed: Vec<usize>,
    /// `indices_freefixed[node]` = row in packed `[free coords…, fixed coords…]`.
    pub indices_freefixed: Vec<usize>,
}

impl Structure {
    pub fn from_network(network: &Network) -> Self {
        let num_nodes = network.num_nodes();
        let num_edges = network.num_edges();
        let connectivity = incidence_matrix(num_edges, num_nodes, &network.edges);

        let mut indices_free = Vec::new();
        let mut indices_fixed = Vec::new();
        for i in 0..num_nodes {
            if network.is_support[i] {
                indices_fixed.push(i);
            } else {
                indices_free.push(i);
            }
        }

        let mut indices_freefixed = vec![0; num_nodes];
        for (pack_i, &node) in indices_free.iter().enumerate() {
            indices_freefixed[node] = pack_i;
        }
        let base = indices_free.len();
        for (pack_i, &node) in indices_fixed.iter().enumerate() {
            indices_freefixed[node] = base + pack_i;
        }

        Self {
            num_nodes,
            num_edges,
            connectivity,
            indices_free,
            indices_fixed,
            indices_freefixed,
        }
    }

    pub fn num_free(&self) -> usize {
        self.indices_free.len()
    }

    pub fn num_fixed(&self) -> usize {
        self.indices_fixed.len()
    }

    /// Row `e`, column `n` of the full incidence matrix.
    #[inline]
    pub fn c(&self, edge: usize, node: usize) -> f64 {
        self.connectivity[edge * self.num_nodes + node]
    }

    /// Submatrix `C_free` with columns `indices_free`.
    pub fn connectivity_free(&self) -> Vec<f64> {
        submatrix_columns(
            &self.connectivity,
            self.num_edges,
            self.num_nodes,
            &self.indices_free,
        )
    }

    /// Submatrix `C_fixed` with columns `indices_fixed`.
    pub fn connectivity_fixed(&self) -> Vec<f64> {
        submatrix_columns(
            &self.connectivity,
            self.num_edges,
            self.num_nodes,
            &self.indices_fixed,
        )
    }
}

/// Directed incidence matching COMPAS `connectivity_matrix` for `(u, v)` edges.
fn incidence_matrix(num_edges: usize, num_nodes: usize, edges: &[(usize, usize)]) -> Vec<f64> {
    let mut c = vec![0.0; num_edges * num_nodes];
    for (e, &(u, v)) in edges.iter().enumerate() {
        c[e * num_nodes + u] = -1.0;
        c[e * num_nodes + v] = 1.0;
    }
    c
}

fn submatrix_columns(
    full: &[f64],
    rows: usize,
    cols: usize,
    col_indices: &[usize],
) -> Vec<f64> {
    let sub_cols = col_indices.len();
    let mut out = vec![0.0; rows * sub_cols];
    for r in 0..rows {
        for (j, &c) in col_indices.iter().enumerate() {
            out[r * sub_cols + j] = full[r * cols + c];
        }
    }
    out
}
