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
pub mod stream;
pub mod symmetric;

pub use arena::DeviceArena;
pub use buffer::Buffer;
pub use collective::{ReduceKind, all_gather, all_reduce, reduce_scatter};
pub use device::Device;
pub use handle::BufferHandle;
pub use stream::{CommandStream, SyncStream};
pub use symmetric::{
    CollectiveError, LocalTransport, Rank, SymmetricBuffer, SymmetricHeap, SymmetricTransport,
};
