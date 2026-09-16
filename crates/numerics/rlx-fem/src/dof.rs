// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fixed and tied degrees of freedom.
//!
//! Constraints are imposed by *elimination*: a fixed node is not an unknown, and
//! a tied node is not an unknown either — it is a signed multiple of the node it
//! follows. The reduced system therefore stays symmetric positive definite,
//! which neither a penalty term nor a Lagrange multiplier block would leave it.
//!
//! Periodic and anti-periodic boundaries are the `+1` and `-1` cases of a tie.
//! For a problem with a repeating structure that is the difference between
//! solving one period and solving all of them, and it is exact rather than an
//! approximation, so it is usually the largest single saving available.

/// What a node contributes to the reduced system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Dof {
    /// Prescribed to zero; contributes nothing.
    Fixed,
    /// An unknown, at this index in the reduced system.
    Free(usize),
    /// Equal to `factor` times the unknown at `leader`.
    Tied {
        /// Reduced-system index this node follows.
        leader: usize,
        /// Multiplier: `+1` for periodic, `-1` for anti-periodic.
        factor: f64,
    },
}

/// Mapping from mesh nodes to reduced-system unknowns.
#[derive(Debug, Clone)]
pub struct DofMap {
    /// One entry per node.
    pub nodes: Vec<Dof>,
    /// Number of unknowns after elimination.
    pub n_free: usize,
}

/// Incremental construction of a [`DofMap`].
#[derive(Debug, Clone)]
pub struct DofMapBuilder {
    fixed: Vec<bool>,
    tie: Vec<Option<(usize, f64)>>,
}

impl DofMap {
    /// Start building a map over `n_nodes` nodes, all initially free.
    pub fn builder(n_nodes: usize) -> DofMapBuilder {
        DofMapBuilder {
            fixed: vec![false; n_nodes],
            tie: vec![None; n_nodes],
        }
    }

    /// Reduced index and multiplier for a node, or `None` if it is fixed.
    #[inline]
    pub fn resolve(&self, node: u32) -> Option<(usize, f64)> {
        match self.nodes[node as usize] {
            Dof::Fixed => None,
            Dof::Free(i) => Some((i, 1.0)),
            Dof::Tied { leader, factor } => Some((leader, factor)),
        }
    }

    /// Contract a per-node covector onto the reduced system.
    ///
    /// The transpose of [`DofMap::expand`], and the operation a right-hand side
    /// needs: a fixed node contributes nothing, and a tied node adds its value,
    /// scaled by its factor, to the node it follows. Objective sensitivities
    /// enter the adjoint system through here.
    pub fn reduce(&self, nodal: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_free];
        for (node, value) in nodal.iter().enumerate() {
            if let Some((index, factor)) = self.resolve(node as u32) {
                out[index] += factor * value;
            }
        }
        out
    }

    /// The reduced unknowns of a nodal state.
    ///
    /// The inverse of [`DofMap::expand`] for a state that already satisfies the
    /// constraints — not the transpose, which is [`DofMap::reduce`]. Reading a
    /// free node's value is all it takes; a tied node carries no information its
    /// leader does not.
    pub fn reduce_state(&self, nodal: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n_free];
        for (node, value) in nodal.iter().enumerate() {
            if let Dof::Free(i) = self.nodes[node] {
                out[i] = *value;
            }
        }
        out
    }

    /// Expand a reduced solution back to one value per node.
    pub fn expand(&self, reduced: &[f64]) -> Vec<f64> {
        self.nodes
            .iter()
            .map(|d| match *d {
                Dof::Fixed => 0.0,
                Dof::Free(i) => reduced[i],
                Dof::Tied { leader, factor } => factor * reduced[leader],
            })
            .collect()
    }
}

impl DofMapBuilder {
    /// Prescribe these nodes to zero.
    pub fn fix(mut self, nodes: &[u32]) -> Self {
        for &n in nodes {
            self.fixed[n as usize] = true;
        }
        self
    }

    /// Tie each node of `followers` to the matching node of `leaders`, so that
    /// the follower's value is `factor` times the leader's.
    ///
    /// # Panics
    ///
    /// If the two slices differ in length.
    pub fn tie(mut self, leaders: &[u32], followers: &[u32], factor: f64) -> Self {
        assert_eq!(
            leaders.len(),
            followers.len(),
            "tied node lists must correspond one to one"
        );
        for (&l, &f) in leaders.iter().zip(followers) {
            let (l, f) = (l as usize, f as usize);
            if l == f {
                continue;
            }
            self.tie[f] = Some((l, factor));
        }
        self
    }

    /// Assign reduced indices.
    ///
    /// A node that is both fixed and tied is fixed, and so is its partner.
    /// Getting that precedence backwards leaves the system singular in exactly
    /// the corner nodes, which is a poor thing to diagnose from a residual.
    pub fn build(mut self) -> DofMap {
        let n = self.fixed.len();

        for i in 0..n {
            if let Some((leader, _)) = self.tie[i]
                && (self.fixed[i] || self.fixed[leader])
            {
                self.fixed[i] = true;
                self.fixed[leader] = true;
                self.tie[i] = None;
            }
        }

        let mut nodes = vec![Dof::Fixed; n];
        let mut n_free = 0;
        for i in 0..n {
            if !self.fixed[i] && self.tie[i].is_none() {
                nodes[i] = Dof::Free(n_free);
                n_free += 1;
            }
        }
        for i in 0..n {
            if let Some((leader, factor)) = self.tie[i] {
                nodes[i] = match nodes[leader] {
                    Dof::Free(m) => Dof::Tied { leader: m, factor },
                    // The leader was itself eliminated, so the follower is too.
                    _ => Dof::Fixed,
                };
            }
        }

        DofMap { nodes, n_free }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_nodes_are_numbered_consecutively() {
        let m = DofMap::builder(5).fix(&[1, 3]).build();
        assert_eq!(m.n_free, 3);
        assert_eq!(m.nodes[0], Dof::Free(0));
        assert_eq!(m.nodes[1], Dof::Fixed);
        assert_eq!(m.nodes[2], Dof::Free(1));
        assert_eq!(m.nodes[4], Dof::Free(2));
    }

    #[test]
    fn a_tie_removes_the_follower_from_the_system() {
        let m = DofMap::builder(4).tie(&[0, 1], &[2, 3], -1.0).build();
        assert_eq!(m.n_free, 2);
        assert_eq!(
            m.nodes[2],
            Dof::Tied {
                leader: 0,
                factor: -1.0
            }
        );
        let expanded = m.expand(&[5.0, 7.0]);
        assert_eq!(expanded, vec![5.0, 7.0, -5.0, -7.0]);
    }

    #[test]
    fn periodic_and_anti_periodic_differ_only_in_sign() {
        let periodic = DofMap::builder(4).tie(&[0, 1], &[2, 3], 1.0).build();
        assert_eq!(periodic.expand(&[2.0, 3.0]), vec![2.0, 3.0, 2.0, 3.0]);
        let anti = DofMap::builder(4).tie(&[0, 1], &[2, 3], -1.0).build();
        assert_eq!(anti.expand(&[2.0, 3.0]), vec![2.0, 3.0, -2.0, -3.0]);
    }

    #[test]
    fn fixing_either_end_of_a_tie_fixes_both() {
        let m = DofMap::builder(4)
            .fix(&[0])
            .tie(&[0, 1], &[2, 3], -1.0)
            .build();
        assert_eq!(m.nodes[0], Dof::Fixed);
        assert_eq!(m.nodes[2], Dof::Fixed);
        // The untouched pair survives.
        assert_eq!(m.nodes[1], Dof::Free(0));
        assert_eq!(
            m.nodes[3],
            Dof::Tied {
                leader: 0,
                factor: -1.0
            }
        );
        assert_eq!(m.n_free, 1);
    }

    #[test]
    fn fixing_the_follower_also_fixes_the_leader() {
        let m = DofMap::builder(4)
            .fix(&[3])
            .tie(&[0, 1], &[2, 3], 1.0)
            .build();
        assert_eq!(m.nodes[1], Dof::Fixed);
        assert_eq!(m.nodes[3], Dof::Fixed);
    }

    #[test]
    fn a_node_tied_to_itself_is_ignored() {
        let m = DofMap::builder(3).tie(&[1], &[1], -1.0).build();
        assert_eq!(m.n_free, 3);
        assert_eq!(m.nodes[1], Dof::Free(1));
    }

    #[test]
    fn reduce_is_the_transpose_of_expand() {
        // The defining property: <reduce(v), r> == <v, expand(r)> for every v
        // and r. If it fails, the adjoint solves the wrong system and the
        // gradients it produces are wrong in a way no residual will reveal.
        let m = DofMap::builder(6)
            .fix(&[0])
            .tie(&[1, 2], &[4, 5], -1.0)
            .build();
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let r: Vec<f64> = (0..m.n_free).map(|i| (i as f64 + 1.0) * 0.7).collect();
        let lhs: f64 = m.reduce(&v).iter().zip(&r).map(|(a, b)| a * b).sum();
        let rhs: f64 = v.iter().zip(m.expand(&r)).map(|(a, b)| a * b).sum();
        assert!((lhs - rhs).abs() < 1e-12, "{lhs} vs {rhs}");
    }

    #[test]
    fn reduce_drops_fixed_nodes_and_folds_tied_ones() {
        let m = DofMap::builder(4).fix(&[0]).tie(&[1], &[2], -1.0).build();
        // Node 0 fixed, node 1 leads, node 2 follows with -1, node 3 free.
        let got = m.reduce(&[9.0, 2.0, 5.0, 1.0]);
        assert_eq!(got[0], 2.0 - 5.0);
    }

    #[test]
    fn resolve_reports_the_multiplier() {
        let m = DofMap::builder(3).fix(&[0]).tie(&[1], &[2], -1.0).build();
        assert_eq!(m.resolve(0), None);
        assert_eq!(m.resolve(1), Some((0, 1.0)));
        assert_eq!(m.resolve(2), Some((0, -1.0)));
    }
}
