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

//! RLX driver layer — devices, arenas, buffers, command streams
//! (plan #58).
//!
//! Borrowed from MAX's three-layer separation: graph (IR) →
//! engine (compiled artifacts, sessions) → driver (devices,
//! buffers). The Rust spelling sits one crate below
//! `rlx-runtime`: this crate owns the *physical* concerns
//! (which device, which buffer slot, which command stream),
//! `rlx-runtime` owns the *logical* engine (Session, CompiledGraph,
//! compile cache).
//!
//! Why split? Three reasons.
//!   1. **Backend symmetry.** rlx-cpu / rlx-metal don't currently
//!      depend on rlx-runtime; before this split they couldn't
//!      reach the `Device` enum without a circular dep. The
//!      `rlx-ir → rlx-driver → backends → rlx-runtime` chain is
//!      strictly one-way.
//!   2. **Testability.** A `Buffer` parity test doesn't need to
//!      pull in the entire compile + execute pipeline.
//!   3. **Future swaps.** Replacing the engine layer (e.g. for
//!      AOT compilation) doesn't touch the driver.
//!
//! `rlx-runtime` re-exports every type here, so existing callers
//! keep working without import changes.

pub mod arena;
pub mod buffer;
pub mod collective;
pub mod device;
pub mod handle;
#[cfg(feature = "iroh")]
pub mod iroh_transport;
#[cfg(feature = "mpi")]
pub mod mpi_transport;
pub mod net;
pub mod node;
pub mod stream;
pub mod symmetric;
pub mod transport;

pub use arena::DeviceArena;
pub use buffer::Buffer;
pub use collective::{ReduceKind, all_gather, all_reduce, reduce_scatter, ring_all_reduce};
pub use device::{
    BackendSupport, Device, DeviceFromStrError, STANDARD_DEVICES, StandardBackends, validate_device,
};
pub use handle::BufferHandle;
#[cfg(feature = "iroh")]
pub use iroh_transport::{
    IrohPeer, IrohTransport, RLX_PIPELINE_ALPN, RelayMap, RelayMode, process_group_from_env,
};
#[cfg(feature = "mpi")]
pub use mpi_transport::MpiTransport;
pub use net::{DEFAULT_HEAP_BYTES, NetTransport, TcpTransport, ThunderboltTransport};
pub use node::{
    Node, Topology, announce_coordinator, discover_coordinator, discover_peers, local_ip,
};
pub use stream::{CommandStream, SyncStream};
pub use symmetric::{
    CollectiveError, LocalTransport, Rank, SymmetricBuffer, SymmetricHeap, SymmetricTransport,
};
pub use transport::{ProcessGroup, ReduceMode, Transport, default_barrier, env_reduce_mode};
