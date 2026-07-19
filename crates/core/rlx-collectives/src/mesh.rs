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

//! Device mesh + named axes — the compositional layer for combining parallelism
//! strategies (DP × TP × PP × EP).
//!
//! A [`Mesh`] lays the global ranks out as a **row-major grid** with named axes,
//! e.g. `["dp", "tp"]` of shape `[2, 2]` maps rank `dp*2 + tp` to coordinate
//! `(dp, tp)`. A collective *along axis A* runs over the ranks that share every
//! *other* coordinate — the sub-group that varies only along A. The mesh
//! resolves, for this rank and an axis:
//!   * [`axis_group_ranks`](Mesh::axis_group_ranks) — the sub-group's members,
//!   * [`axis_rank`](Mesh::axis_rank) / [`axis_size`](Mesh::axis_size) — this
//!     rank's index and the group size, and
//!   * [`group`](Mesh::group) — a stable id to register that sub-group's
//!     [`rlx_driver::ProcessGroup`] under (distinct sub-groups along one axis
//!     get distinct ids).
//!
//! Wire it once — build each axis's sub-group `ProcessGroup` and
//! `register_group(mesh.group("tp"), tp_group)` — then pass `mesh.group("tp")`
//! as the `group_id` to any collective (`all_reduce`, `reduce_scatter`, …) to
//! run it along that axis. Nothing else changes; the mesh is pure indexing over
//! the existing collectives.

/// A named, row-major layout of the global ranks.
#[derive(Clone, Debug)]
pub struct Mesh {
    axes: Vec<String>,
    shape: Vec<usize>,
    rank: u32,
    coords: Vec<usize>,
    base_id: u64,
}

impl Mesh {
    /// Build the mesh view for `rank`. `axes` names each dimension, `shape`
    /// gives its size (row-major, last axis fastest), and `base_id` seeds the
    /// per-axis sub-group ids. `axes.len() == shape.len()` and
    /// `rank < product(shape)`.
    pub fn new(axes: &[&str], shape: &[usize], rank: u32, base_id: u64) -> Self {
        assert_eq!(axes.len(), shape.len(), "mesh: axes/shape rank mismatch");
        let total: usize = shape.iter().product();
        assert!(
            (rank as usize) < total,
            "mesh: rank {rank} outside {total} ranks"
        );
        let mut coords = vec![0usize; shape.len()];
        let mut r = rank as usize;
        for i in (0..shape.len()).rev() {
            coords[i] = r % shape[i];
            r /= shape[i];
        }
        Self {
            axes: axes.iter().map(|s| s.to_string()).collect(),
            shape: shape.to_vec(),
            rank,
            coords,
            base_id,
        }
    }

    /// This rank's global id.
    pub fn rank(&self) -> u32 {
        self.rank
    }

    /// This rank's coordinate along every axis.
    pub fn coords(&self) -> &[usize] {
        &self.coords
    }

    fn axis_index(&self, axis: &str) -> usize {
        self.axes
            .iter()
            .position(|a| a == axis)
            .unwrap_or_else(|| panic!("mesh: unknown axis '{axis}'"))
    }

    fn coords_to_rank(&self, c: &[usize]) -> u32 {
        let mut r = 0usize;
        for i in 0..self.shape.len() {
            r = r * self.shape[i] + c[i];
        }
        r as u32
    }

    /// Size of `axis` (the number of ranks in any group along it).
    pub fn axis_size(&self, axis: &str) -> usize {
        self.shape[self.axis_index(axis)]
    }

    /// This rank's index within its sub-group along `axis`.
    pub fn axis_rank(&self, axis: &str) -> u32 {
        self.coords[self.axis_index(axis)] as u32
    }

    /// The global ranks in this rank's sub-group along `axis` (they share every
    /// other coordinate), ordered by their `axis` index.
    pub fn axis_group_ranks(&self, axis: &str) -> Vec<u32> {
        let a = self.axis_index(axis);
        (0..self.shape[a])
            .map(|k| {
                let mut c = self.coords.clone();
                c[a] = k;
                self.coords_to_rank(&c)
            })
            .collect()
    }

    /// A stable id for this rank's sub-group along `axis`: `base_id` offset by
    /// the axis plus the flattened *other* coordinates, so every distinct
    /// sub-group along one axis gets a distinct id and ranks in the same
    /// sub-group agree on it.
    pub fn group(&self, axis: &str) -> u64 {
        let a = self.axis_index(axis);
        let mut other = 0u64;
        let mut mul = 1u64;
        for i in 0..self.shape.len() {
            if i == a {
                continue;
            }
            other += self.coords[i] as u64 * mul;
            mul *= self.shape[i] as u64;
        }
        // Axis stride large enough that axes never alias for realistic meshes.
        self.base_id + (a as u64) * 1_000_000 + other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{all_reduce, register_group, unregister_group};
    use rlx_driver::{NetTransport, ProcessGroup};
    use rlx_ir::{DType, Graph, Shape};
    use rlx_runtime::{Device, Session};
    use std::net::{Ipv4Addr, SocketAddr, TcpListener};
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn mesh_2x2_indexing() {
        // axes [dp, tp], shape [2, 2] → rank = dp*2 + tp.
        let m = |r: u32| Mesh::new(&["dp", "tp"], &[2, 2], r, 100);
        assert_eq!(m(0).coords(), &[0, 0]);
        assert_eq!(m(3).coords(), &[1, 1]);

        // tp sub-groups: {0,1} (dp=0) and {2,3} (dp=1).
        assert_eq!(m(0).axis_group_ranks("tp"), vec![0, 1]);
        assert_eq!(m(3).axis_group_ranks("tp"), vec![2, 3]);
        // dp sub-groups: {0,2} (tp=0) and {1,3} (tp=1).
        assert_eq!(m(0).axis_group_ranks("dp"), vec![0, 2]);
        assert_eq!(m(1).axis_group_ranks("dp"), vec![1, 3]);

        assert_eq!(m(0).axis_rank("tp"), 0);
        assert_eq!(m(3).axis_rank("tp"), 1);

        // Ranks in the same sub-group share an id; different sub-groups differ.
        assert_eq!(m(0).group("tp"), m(1).group("tp")); // {0,1}
        assert_eq!(m(2).group("tp"), m(3).group("tp")); // {2,3}
        assert_ne!(m(0).group("tp"), m(2).group("tp"));
        assert_eq!(m(0).group("dp"), m(2).group("dp")); // {0,2}
        assert_ne!(m(0).group("dp"), m(0).group("tp")); // axes don't alias
    }

    /// End-to-end: 4 ranks in a 2×2 mesh, all-reduce **along the `tp` axis** must
    /// reduce only within each tp sub-group ({0,1} and {2,3}), not across all 4.
    #[test]
    fn mesh_all_reduce_along_axis() {
        let n = 4usize;
        // Two tp-groups of 2 ranks each; set up a NetTransport per group.
        // Global rank r → tp-group (r/2), within-group rank (r%2).
        let listeners: Vec<Vec<TcpListener>> = (0..2)
            .map(|_| {
                (0..2)
                    .map(|_| TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap())
                    .collect()
            })
            .collect();
        let addrs: Vec<Vec<SocketAddr>> = listeners
            .iter()
            .map(|g| g.iter().map(|l| l.local_addr().unwrap()).collect())
            .collect();

        // Flatten listeners keyed by global rank.
        let mut per_rank: Vec<(usize, TcpListener, Vec<SocketAddr>)> = Vec::new();
        for (g, group) in listeners.into_iter().enumerate() {
            for (within, l) in group.into_iter().enumerate() {
                per_rank.push((g, l, addrs[g].clone()));
                let _ = within;
            }
        }

        let handles: Vec<_> = per_rank
            .into_iter()
            .enumerate()
            .map(|(global_rank, (_g, listener, group_addrs))| {
                thread::spawn(move || {
                    crate::register();
                    let global_rank = global_rank as u32;
                    let mesh = Mesh::new(&["dp", "tp"], &[2, 2], global_rank, 700);
                    let within = mesh.axis_rank("tp"); // this rank's index in its tp group
                    let t = NetTransport::from_listener(within, 2, listener, group_addrs, 1 << 20)
                        .unwrap();
                    // The mesh picks the sub-group *topology* (which peers form the
                    // `tp` transport, via `axis_group_ranks`). In real deployment
                    // `mesh.group("tp")` is the registry key (registries are
                    // per-process); here all 4 ranks share one process, so we key
                    // by a per-rank-unique local id — each rank resolves its own
                    // ProcessGroup, and its transport handles the peers.
                    let reg_id = 900 + global_rank as u64;
                    register_group(reg_id, Arc::new(ProcessGroup::new(Arc::new(t))));

                    // Each rank contributes [global_rank + 1; n].
                    let mut g = Graph::new("mesh_ar");
                    let x = g.input("x", Shape::new(&[n], DType::F32));
                    let y = all_reduce(&mut g, x, reg_id);
                    g.set_outputs(vec![y]);
                    let mut c = Session::new(Device::Cpu).compile(g);
                    let data = vec![(global_rank + 1) as f32; n];
                    let res = c.run(&[("x", data.as_slice())]);
                    unregister_group(reg_id);
                    res.into_iter().next().unwrap()
                })
            })
            .collect();

        let outs: Vec<Vec<f32>> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        // tp-group {0,1}: sum(1,2)=3. tp-group {2,3}: sum(3,4)=7. NOT 10.
        assert_eq!(outs[0], vec![3.0; n], "rank 0 (tp-group {{0,1}})");
        assert_eq!(outs[1], vec![3.0; n], "rank 1 (tp-group {{0,1}})");
        assert_eq!(outs[2], vec![7.0; n], "rank 2 (tp-group {{2,3}})");
        assert_eq!(outs[3], vec![7.0; n], "rank 3 (tp-group {{2,3}})");
    }
}
