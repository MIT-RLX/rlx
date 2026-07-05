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

//! [`MpiTransport`] — a [`Transport`](crate::transport::Transport) backed by
//! the pure-Rust `mpi` crate (the C-free `mpi-no-c` implementation, API-
//! compatible with rsmpi). Gated behind the `mpi` cargo feature.
//!
//! RLX already ships a native, zero-dependency [`NetTransport`](crate::net::NetTransport)
//! (TCP full-mesh). This adapter is the *opt-in alternative*: it lets the same
//! [`ProcessGroup`](crate::transport::ProcessGroup) collectives ride on MPI so
//! a job can be launched and bootstrapped with the standard `mpirun` /
//! `mpiexec` (hostfile + rank assignment) instead of hand-wiring peer IPs.
//! Everything above the [`Transport`](crate::transport::Transport) trait —
//! `all_reduce` / `all_gather` / `broadcast` / pipeline handoff — is unchanged.
//!
//! ```no_run
//! # #[cfg(feature = "mpi")]
//! # fn demo() {
//! use rlx_driver::{MpiTransport, ProcessGroup};
//! use std::sync::Arc;
//!
//! let universe = mpi::initialize().unwrap();
//! // `universe` owns MPI init/finalize — keep it alive for the whole job.
//! let group = ProcessGroup::new(Arc::new(MpiTransport::new(universe.world())));
//! // group.all_reduce(...) now runs over MPI.
//! # }
//! ```

use crate::symmetric::CollectiveError;
use crate::transport::Transport;
use mpi::collective::CommunicatorCollectives;
use mpi::point_to_point::{Destination, Source};
use mpi::topology::{Communicator, SimpleCommunicator};

/// A two-sided [`Transport`](crate::transport::Transport) over an MPI
/// communicator (`MPI_COMM_WORLD` by default).
///
/// The wrapped [`SimpleCommunicator`] is `Arc`-backed and `Send + Sync`, so the
/// transport can be shared across threads like the native ones. The owning
/// [`mpi::environment::Universe`] governs `MPI_Init` / `MPI_Finalize` and must
/// outlive this transport (standard MPI lifetime, mirroring rsmpi).
pub struct MpiTransport {
    world: SimpleCommunicator,
}

impl MpiTransport {
    /// Wrap a communicator (typically `universe.world()`).
    pub fn new(world: SimpleCommunicator) -> Self {
        Self { world }
    }

    /// The underlying MPI communicator.
    pub fn communicator(&self) -> &SimpleCommunicator {
        &self.world
    }
}

/// Fold an RLX `u32` tag into MPI's tag space. MPI tags are non-negative
/// `i32`; RLX reserves the high range (`>= TAG_RESERVED_BASE`, i.e. bit 31
/// set) for barrier/collective traffic, so masking bit 31 keeps every tag
/// non-negative (never `MPI_ANY_TAG == -1`). The mask is applied identically
/// on send and receive, so matching is preserved; it is injective over RLX's
/// actual tag usage (a handful of reserved tags plus small user tags — no two
/// differ only in bit 31).
#[inline]
fn map_tag(tag: u32) -> mpi::Tag {
    (tag & 0x7FFF_FFFF) as mpi::Tag
}

impl Transport for MpiTransport {
    fn rank(&self) -> u32 {
        self.world.rank() as u32
    }

    fn world_size(&self) -> u32 {
        self.world.size() as u32
    }

    fn send_bytes(&self, to: u32, tag: u32, bytes: &[u8]) -> Result<(), CollectiveError> {
        self.world
            .process_at_rank(to as mpi::Rank)
            .send_with_tag(bytes, map_tag(tag));
        Ok(())
    }

    fn recv_bytes(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError> {
        let (buf, _status) = self
            .world
            .process_at_rank(from as mpi::Rank)
            .receive_vec_with_tag::<u8>(map_tag(tag));
        Ok(buf)
    }

    /// Native MPI barrier — collective over the whole communicator, faster than
    /// the default gather-to-root rendezvous.
    fn barrier(&self) -> Result<(), CollectiveError> {
        self.world.barrier();
        Ok(())
    }
}
