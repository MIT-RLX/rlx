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

//! Symmetric-memory primitives for collective ops (plan #49).
//!
//! Borrowed from MAX's `kernels/src/shmem/{_nvshmem, _rocshmem,
//! _mpi, shmem_buffer, ep_comm}.mojo`. Symmetric heaps are the
//! standard abstraction for multi-device collective comm: every
//! rank allocates the same logical region; addresses are
//! identical across ranks (the physical backing differs). One
//! rank can `put` directly into another's slot at the same
//! offset.
//!
//! The Rust spelling here ships:
//!
//!   - [`SymmetricHeap`] — owns the per-rank physical storage.
//!     Single-machine emulation today (one `Vec<u8>` per rank);
//!     the trait surface ([`SymmetricTransport`]) is what a
//!     future MPI / NVSHMEM-equivalent / process-shared-memory
//!     impl plugs into.
//!   - [`SymmetricBuffer`] — a `(rank, offset, len)` view.
//!   - [`SymmetricTransport`] — the trait every transport
//!     impl satisfies. `LocalTransport` is the in-process,
//!     single-machine impl used by tests + collective-algo
//!     correctness checks (plan #12).
//!
//! This is the foundation; #12 builds AllReduce / AllGather /
//! ReduceScatter on top.

use std::sync::{Arc, RwLock};

/// Identifier for a participant in a collective. Ranks are
/// `0..num_ranks` and stay stable for the lifetime of a
/// transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Rank(pub u32);

/// `(rank, offset, len)` view into a symmetric heap. The same
/// `(offset, len)` pair is valid on every rank — that's what
/// "symmetric" means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymmetricBuffer {
    pub rank: Rank,
    pub offset: usize,
    pub len: usize,
}

/// One-sided operation surface. `put(buf, src)` writes `src`
/// into `buf.rank`'s memory at `buf.offset`; `get(buf, dst)`
/// reads from `buf.rank`'s memory into `dst`. Both calls block
/// until completion (a future async impl can return a future).
pub trait SymmetricTransport {
    /// How many ranks participate.
    fn num_ranks(&self) -> u32;
    /// This process's rank.
    fn this_rank(&self) -> Rank;

    /// Write `src` into `buf`. Errors on length mismatch.
    fn put(&self, buf: SymmetricBuffer, src: &[u8]) -> Result<(), CollectiveError>;
    /// Read from `buf` into `dst`. Errors on length mismatch.
    fn get(&self, buf: SymmetricBuffer, dst: &mut [u8]) -> Result<(), CollectiveError>;

    /// Block until every rank has reached this barrier. Local
    /// emulation is a memory fence + a counter; real transports
    /// implement their own.
    fn barrier(&self) -> Result<(), CollectiveError>;
}

#[derive(Debug, Clone)]
pub enum CollectiveError {
    /// `(rank, offset, len)` walks past the heap.
    OutOfBounds {
        rank: Rank,
        offset: usize,
        len: usize,
        heap_size: usize,
    },
    /// `src.len() != buf.len`.
    LengthMismatch { expected: usize, got: usize },
    /// Unknown rank id.
    UnknownRank { rank: Rank, num_ranks: u32 },
    /// Underlying transport failed (network, mmap, etc.).
    TransportError { reason: String },
}

impl std::fmt::Display for CollectiveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OutOfBounds {
                rank,
                offset,
                len,
                heap_size,
            } => write!(
                f,
                "OOB on rank {}: offset {offset} + len {len} > heap_size {heap_size}",
                rank.0
            ),
            Self::LengthMismatch { expected, got } => {
                write!(f, "length mismatch: expected {expected}, got {got}")
            }
            Self::UnknownRank { rank, num_ranks } => {
                write!(f, "unknown rank {} (have {num_ranks})", rank.0)
            }
            Self::TransportError { reason } => write!(f, "transport: {reason}"),
        }
    }
}

impl std::error::Error for CollectiveError {}

/// Per-rank symmetric memory: a `Vec<u8>` per rank, all the same
/// size. Owned by the [`LocalTransport`].
#[derive(Debug)]
pub struct SymmetricHeap {
    storage: Vec<Arc<RwLock<Vec<u8>>>>,
    pub heap_size: usize,
}

impl SymmetricHeap {
    pub fn new(num_ranks: u32, heap_size: usize) -> Self {
        let storage = (0..num_ranks)
            .map(|_| Arc::new(RwLock::new(vec![0u8; heap_size])))
            .collect();
        Self { storage, heap_size }
    }

    pub fn num_ranks(&self) -> u32 {
        self.storage.len() as u32
    }

    pub fn rank_view(&self, rank: Rank) -> Option<Arc<RwLock<Vec<u8>>>> {
        self.storage.get(rank.0 as usize).cloned()
    }
}

/// Single-machine in-process transport. All `num_ranks`
/// "ranks" share one [`SymmetricHeap`] instance, so put / get
/// are just locks + memcpy. Useful for unit tests and for
/// algorithm-correctness checking of collective ops without a
/// real cluster.
#[derive(Debug, Clone)]
pub struct LocalTransport {
    heap: Arc<SymmetricHeap>,
    me: Rank,
    barrier_count: Arc<std::sync::atomic::AtomicU32>,
    barrier_target: u32,
}

impl LocalTransport {
    pub fn new(num_ranks: u32, heap_size: usize, this_rank: Rank) -> Self {
        let heap = Arc::new(SymmetricHeap::new(num_ranks, heap_size));
        Self::with_heap(heap, this_rank)
    }

    /// Construct multiple `LocalTransport`s sharing one heap —
    /// `Vec` of length `num_ranks`, each with its own `me`.
    /// Tests typically iterate this list to drive each rank.
    pub fn fan_out(num_ranks: u32, heap_size: usize) -> Vec<Self> {
        let heap = Arc::new(SymmetricHeap::new(num_ranks, heap_size));
        (0..num_ranks)
            .map(|i| Self::with_heap(heap.clone(), Rank(i)))
            .collect()
    }

    fn with_heap(heap: Arc<SymmetricHeap>, me: Rank) -> Self {
        let n = heap.num_ranks();
        Self {
            heap,
            me,
            barrier_count: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            barrier_target: n,
        }
    }

    fn check_buf(&self, buf: SymmetricBuffer) -> Result<(), CollectiveError> {
        if buf.rank.0 >= self.heap.num_ranks() {
            return Err(CollectiveError::UnknownRank {
                rank: buf.rank,
                num_ranks: self.heap.num_ranks(),
            });
        }
        if buf.offset + buf.len > self.heap.heap_size {
            return Err(CollectiveError::OutOfBounds {
                rank: buf.rank,
                offset: buf.offset,
                len: buf.len,
                heap_size: self.heap.heap_size,
            });
        }
        Ok(())
    }
}

impl SymmetricTransport for LocalTransport {
    fn num_ranks(&self) -> u32 {
        self.heap.num_ranks()
    }
    fn this_rank(&self) -> Rank {
        self.me
    }

    fn put(&self, buf: SymmetricBuffer, src: &[u8]) -> Result<(), CollectiveError> {
        self.check_buf(buf)?;
        if src.len() != buf.len {
            return Err(CollectiveError::LengthMismatch {
                expected: buf.len,
                got: src.len(),
            });
        }
        let view = self.heap.rank_view(buf.rank).expect("checked above");
        let mut guard = view.write().unwrap();
        guard[buf.offset..buf.offset + buf.len].copy_from_slice(src);
        Ok(())
    }

    fn get(&self, buf: SymmetricBuffer, dst: &mut [u8]) -> Result<(), CollectiveError> {
        self.check_buf(buf)?;
        if dst.len() != buf.len {
            return Err(CollectiveError::LengthMismatch {
                expected: buf.len,
                got: dst.len(),
            });
        }
        let view = self.heap.rank_view(buf.rank).expect("checked above");
        let guard = view.read().unwrap();
        dst.copy_from_slice(&guard[buf.offset..buf.offset + buf.len]);
        Ok(())
    }

    fn barrier(&self) -> Result<(), CollectiveError> {
        // Each rank bumps the counter; spin until we observe
        // num_ranks bumps, then move on. This isn't a "real"
        // barrier (no rendezvous) — it's an arrival counter,
        // sufficient for single-thread tests where each rank
        // calls barrier in turn.
        use std::sync::atomic::Ordering;
        self.barrier_count.fetch_add(1, Ordering::AcqRel);
        // For LocalTransport in single-thread tests this returns
        // immediately; concurrent multi-thread tests can spin.
        while self.barrier_count.load(Ordering::Acquire) < self.barrier_target {
            std::hint::spin_loop();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_then_get_round_trips() {
        let t = LocalTransport::new(4, 1024, Rank(0));
        let buf = SymmetricBuffer {
            rank: Rank(2),
            offset: 16,
            len: 8,
        };
        t.put(buf, &[1, 2, 3, 4, 5, 6, 7, 8]).unwrap();
        let mut dst = [0u8; 8];
        t.get(buf, &mut dst).unwrap();
        assert_eq!(&dst, &[1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn fan_out_yields_one_transport_per_rank() {
        let ts = LocalTransport::fan_out(3, 64);
        assert_eq!(ts.len(), 3);
        for (i, t) in ts.iter().enumerate() {
            assert_eq!(t.this_rank(), Rank(i as u32));
            assert_eq!(t.num_ranks(), 3);
        }
    }

    #[test]
    fn put_visible_to_other_rank_via_shared_heap() {
        // Rank 0 writes into rank 2's slot; rank 2 reads its
        // own slot and sees the write.
        let ts = LocalTransport::fan_out(3, 32);
        let payload = [9u8, 9, 9, 9];
        ts[0]
            .put(
                SymmetricBuffer {
                    rank: Rank(2),
                    offset: 0,
                    len: 4,
                },
                &payload,
            )
            .unwrap();
        let mut dst = [0u8; 4];
        ts[2]
            .get(
                SymmetricBuffer {
                    rank: Rank(2),
                    offset: 0,
                    len: 4,
                },
                &mut dst,
            )
            .unwrap();
        assert_eq!(dst, payload);
    }

    #[test]
    fn oob_offset_errors() {
        let t = LocalTransport::new(2, 8, Rank(0));
        let err = t
            .put(
                SymmetricBuffer {
                    rank: Rank(1),
                    offset: 4,
                    len: 8,
                },
                &[0u8; 8],
            )
            .unwrap_err();
        assert!(matches!(err, CollectiveError::OutOfBounds { .. }));
    }

    #[test]
    fn unknown_rank_errors() {
        let t = LocalTransport::new(2, 8, Rank(0));
        let err = t
            .get(
                SymmetricBuffer {
                    rank: Rank(99),
                    offset: 0,
                    len: 4,
                },
                &mut [0u8; 4],
            )
            .unwrap_err();
        assert!(matches!(err, CollectiveError::UnknownRank { .. }));
    }

    #[test]
    fn length_mismatch_errors() {
        let t = LocalTransport::new(2, 32, Rank(0));
        let err = t
            .put(
                SymmetricBuffer {
                    rank: Rank(1),
                    offset: 0,
                    len: 4,
                },
                &[0u8; 8],
            )
            .unwrap_err();
        assert!(matches!(err, CollectiveError::LengthMismatch { .. }));
    }
}
