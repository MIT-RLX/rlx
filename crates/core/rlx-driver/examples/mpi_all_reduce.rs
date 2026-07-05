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

//! `all_reduce` across MPI ranks via [`rlx_driver::MpiTransport`] — the same
//! [`rlx_driver::ProcessGroup`] collective, riding the pure-Rust `mpi` backend
//! instead of the native `NetTransport`.
//!
//! Requires the `mpi` feature. Runs as a singleton (world size 1) when launched
//! directly, or across N ranks under the bundled launcher:
//!
//! ```text
//! cargo build -p rlx-driver --features mpi --example mpi_all_reduce
//!
//! # single process (world size 1) — quick sanity check:
//! ./target/debug/examples/mpi_all_reduce
//!
//! # N ranks via mpi-no-c's launcher (build `mpiexec` once in that repo):
//! mpiexec -n 4 ./target/debug/examples/mpi_all_reduce
//! ```

use rlx_driver::{MpiTransport, ProcessGroup, ReduceKind};
use std::sync::Arc;

fn main() {
    // `universe` owns MPI init/finalize — keep it alive for the whole run.
    let universe = mpi::initialize().expect("MPI already initialized");
    let group = ProcessGroup::new(Arc::new(MpiTransport::new(universe.world())));

    let rank = group.rank();
    let world = group.world_size();

    // Rank r contributes [r+1, r+1, r+1]; a sum all-reduce yields
    // Σ_{r=0}^{world-1}(r+1) = world*(world+1)/2 in every slot on every rank.
    let mut data = vec![(rank + 1) as f32; 3];
    group
        .all_reduce(&mut data, ReduceKind::Sum)
        .expect("all_reduce failed");

    let expected = (world * (world + 1) / 2) as f32;
    println!("rank {rank}/{world}: all_reduce(sum) = {data:?}  (expected {expected} in each slot)");
    assert!(
        data.iter().all(|&x| (x - expected).abs() < 1e-4),
        "rank {rank}: all_reduce mismatch (got {data:?}, want {expected})"
    );

    // Native MPI barrier so rank 0's summary prints after everyone verified.
    group.barrier().expect("barrier failed");
    if rank == 0 {
        println!("OK — all {world} rank(s) agree on the reduced value.");
    }
}
