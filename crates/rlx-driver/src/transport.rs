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

//! Two-sided point-to-point transport + process group (plan #12/#49,
//! multi-node extension).
//!
//! [`SymmetricTransport`](crate::symmetric::SymmetricTransport) models
//! *one-sided* `put`/`get` — the right shape for RDMA and for the
//! tensor-parallel all-reduce that lives *inside* a compiled graph.
//! Pipeline parallelism wants the complementary *two-sided* surface:
//! rank `r` hands its hidden state to rank `r-1` with a matched
//! `send`/`recv`. This module adds that surface ([`Transport`]) plus a
//! [`ProcessGroup`] that layers tensor-shaped collectives
//! (`all_reduce` / `all_gather` / `broadcast`) on top.
//!
//! The collectives here are the *gather-to-root* family: simple,
//! deadlock-free over any correct [`Transport`], and good enough for
//! the small rank counts (≤ 8) a few Thunderbolt-linked machines hit.
//! Bandwidth-optimal ring variants are a later optimization — the
//! `ProcessGroup` API does not change when they land.
//!
//! Three transports implement [`Transport`]:
//!   - [`TcpTransport`](crate::net::TcpTransport) — portable TCP, works
//!     over Ethernet and the Thunderbolt Bridge IP link.
//!   - [`ThunderboltTransport`](crate::net::ThunderboltTransport) —
//!     TCP pinned to the Thunderbolt interface, also exposing the
//!     symmetric one-sided heap.
//!   - `MlxTransport` (in `rlx-mlx`) — delegates to MLX's `jaccl`
//!     distributed backend over FFI.

use crate::collective::ReduceKind;
use crate::symmetric::CollectiveError;
use std::sync::Arc;

/// Tags `>= TAG_RESERVED_BASE` are reserved for the collective and
/// barrier machinery. User point-to-point traffic (pipeline-parallel
/// hidden states) must use tags below this.
pub const TAG_RESERVED_BASE: u32 = 0xFFF0_0000;
const TAG_BARRIER: u32 = TAG_RESERVED_BASE;
const TAG_ALL_REDUCE: u32 = TAG_RESERVED_BASE + 1;
const TAG_ALL_GATHER: u32 = TAG_RESERVED_BASE + 2;
const TAG_BROADCAST: u32 = TAG_RESERVED_BASE + 3;

/// Two-sided, tagged, byte-oriented point-to-point transport between
/// `world_size` ranks.
///
/// Implementations must be safe to share across threads (`Send +
/// Sync`) and must let a reader on one rank pull a message that a
/// writer on another rank pushed with the same `(src, tag)` — the
/// `send`/`recv` are *matched*, like MPI. `recv` learns the payload
/// length from the wire, so callers need not know it in advance.
pub trait Transport: Send + Sync {
    /// This process's rank in `0..world_size`.
    fn rank(&self) -> u32;
    /// Total number of participating ranks.
    fn world_size(&self) -> u32;

    /// Send `bytes` to rank `to`, tagged `tag`. Returns once the bytes
    /// are handed to the transport (not necessarily delivered).
    fn send_bytes(&self, to: u32, tag: u32, bytes: &[u8]) -> Result<(), CollectiveError>;

    /// Block until a message with matching `(from, tag)` arrives, and
    /// return its payload.
    fn recv_bytes(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError>;

    /// Block until every rank has reached this barrier.
    ///
    /// Default is a gather-to-root rendezvous over `send`/`recv`: every
    /// non-root sends a token to rank 0 and waits for the release; rank
    /// 0 collects all tokens then releases everyone. Lockstep is
    /// preserved across repeated barriers because a non-root cannot
    /// enter the next barrier until it observes the current release.
    fn barrier(&self) -> Result<(), CollectiveError> {
        default_barrier(self)
    }
}

/// Gather-to-root barrier usable by any [`Transport`]. Pulled out as a
/// free function so trait impls can reuse it from an overridden
/// `barrier` if they want to wrap it.
pub fn default_barrier<T: Transport + ?Sized>(t: &T) -> Result<(), CollectiveError> {
    let n = t.world_size();
    if n <= 1 {
        return Ok(());
    }
    let me = t.rank();
    if me == 0 {
        for r in 1..n {
            t.recv_bytes(r, TAG_BARRIER)?;
        }
        for r in 1..n {
            t.send_bytes(r, TAG_BARRIER, &[1u8])?;
        }
    } else {
        t.send_bytes(0, TAG_BARRIER, &[1u8])?;
        t.recv_bytes(0, TAG_BARRIER)?;
    }
    Ok(())
}

/// Little-endian flatten of an `f32` slice. Apple Silicon and x86 are
/// both little-endian; we fix LE on the wire so a future big-endian
/// rank would still interoperate.
fn f32_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() * 4);
    for &x in data {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

/// Inverse of [`f32_to_le_bytes`]. Errors if `bytes.len()` is not a
/// multiple of 4.
fn le_bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, CollectiveError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(CollectiveError::TransportError {
            reason: format!("recv payload {} bytes is not a multiple of 4", bytes.len()),
        });
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn combine(op: ReduceKind, a: f32, b: f32) -> f32 {
    match op {
        ReduceKind::Sum | ReduceKind::Mean => a + b,
        ReduceKind::Max => a.max(b),
        ReduceKind::Min => a.min(b),
    }
}

fn finalize(op: ReduceKind, acc: f32, n: u32) -> f32 {
    match op {
        ReduceKind::Mean => acc / (n as f32),
        _ => acc,
    }
}

/// A handle to the collective-communication world: a rank, a size, and
/// the transport that connects them.
///
/// Wraps an `Arc<dyn Transport>` so the same group can be cloned into
/// the generator loop, the weight loader, and (later) per-layer
/// collective ops without re-establishing connections.
#[derive(Clone)]
pub struct ProcessGroup {
    transport: Arc<dyn Transport>,
}

impl ProcessGroup {
    pub fn new(transport: Arc<dyn Transport>) -> Self {
        Self { transport }
    }

    pub fn rank(&self) -> u32 {
        self.transport.rank()
    }

    pub fn world_size(&self) -> u32 {
        self.transport.world_size()
    }

    /// `true` on the lowest rank — the canonical place to print, sample,
    /// or own the final logits in a pipeline.
    pub fn is_leader(&self) -> bool {
        self.rank() == 0
    }

    pub fn transport(&self) -> &Arc<dyn Transport> {
        &self.transport
    }

    pub fn barrier(&self) -> Result<(), CollectiveError> {
        self.transport.barrier()
    }

    // ---- point-to-point (pipeline parallel) --------------------------

    /// Send a hidden-state (or any) tensor to `to`. `tag` must be below
    /// [`TAG_RESERVED_BASE`] (those tags are reserved for collectives).
    pub fn send_f32(&self, to: u32, tag: u32, data: &[f32]) -> Result<(), CollectiveError> {
        debug_assert!(tag < TAG_RESERVED_BASE, "tag collides with collective tags");
        self.send_f32_tagged(to, tag, data)
    }

    /// Receive a tensor from `from`. Length is taken from the wire.
    pub fn recv_f32(&self, from: u32, tag: u32) -> Result<Vec<f32>, CollectiveError> {
        debug_assert!(tag < TAG_RESERVED_BASE, "tag collides with collective tags");
        self.recv_f32_tagged(from, tag)
    }

    /// Internal f32 send with no reserved-tag guard — collectives use
    /// the reserved tags legitimately.
    fn send_f32_tagged(&self, to: u32, tag: u32, data: &[f32]) -> Result<(), CollectiveError> {
        self.transport.send_bytes(to, tag, &f32_to_le_bytes(data))
    }

    fn recv_f32_tagged(&self, from: u32, tag: u32) -> Result<Vec<f32>, CollectiveError> {
        le_bytes_to_f32(&self.transport.recv_bytes(from, tag)?)
    }

    // ---- collectives (tensor parallel) -------------------------------

    /// In-place all-reduce: on return every rank holds
    /// `op({each rank's input})`. All ranks must pass the same length.
    ///
    /// Gather-to-root: non-roots ship their vector to rank 0, rank 0
    /// reduces and ships the result back. O(n) at the root — fine for a
    /// handful of nodes; swap in ring-reduce later without touching
    /// callers.
    pub fn all_reduce(&self, data: &mut [f32], op: ReduceKind) -> Result<(), CollectiveError> {
        let n = self.world_size();
        if n <= 1 {
            for v in data.iter_mut() {
                *v = finalize(op, *v, n.max(1));
            }
            return Ok(());
        }
        let elems = data.len();
        if self.rank() == 0 {
            let mut acc = data.to_vec();
            for r in 1..n {
                let other = self.recv_f32_tagged(r, TAG_ALL_REDUCE)?;
                if other.len() != elems {
                    return Err(CollectiveError::LengthMismatch {
                        expected: elems,
                        got: other.len(),
                    });
                }
                for i in 0..elems {
                    acc[i] = combine(op, acc[i], other[i]);
                }
            }
            for v in acc.iter_mut() {
                *v = finalize(op, *v, n);
            }
            data.copy_from_slice(&acc);
            for r in 1..n {
                self.send_f32_tagged(r, TAG_ALL_REDUCE, &acc)?;
            }
        } else {
            self.send_f32_tagged(0, TAG_ALL_REDUCE, data)?;
            let res = self.recv_f32_tagged(0, TAG_ALL_REDUCE)?;
            if res.len() != elems {
                return Err(CollectiveError::LengthMismatch {
                    expected: elems,
                    got: res.len(),
                });
            }
            data.copy_from_slice(&res);
        }
        Ok(())
    }

    /// All-gather: every rank contributes `local` (same length on every
    /// rank) and receives the concatenation in rank order, length
    /// `world_size * local.len()`.
    pub fn all_gather(&self, local: &[f32]) -> Result<Vec<f32>, CollectiveError> {
        let n = self.world_size();
        let len = local.len();
        if n <= 1 {
            return Ok(local.to_vec());
        }
        if self.rank() == 0 {
            let mut out = vec![0f32; n as usize * len];
            out[..len].copy_from_slice(local);
            for r in 1..n {
                let chunk = self.recv_f32_tagged(r, TAG_ALL_GATHER)?;
                if chunk.len() != len {
                    return Err(CollectiveError::LengthMismatch {
                        expected: len,
                        got: chunk.len(),
                    });
                }
                let start = r as usize * len;
                out[start..start + len].copy_from_slice(&chunk);
            }
            for r in 1..n {
                self.send_f32_tagged(r, TAG_ALL_GATHER, &out)?;
            }
            Ok(out)
        } else {
            self.send_f32_tagged(0, TAG_ALL_GATHER, local)?;
            let out = self.recv_f32_tagged(0, TAG_ALL_GATHER)?;
            if out.len() != n as usize * len {
                return Err(CollectiveError::LengthMismatch {
                    expected: n as usize * len,
                    got: out.len(),
                });
            }
            Ok(out)
        }
    }

    /// Broadcast `data` from `root` to every rank. On non-root ranks
    /// `data` is overwritten; its length must already match the root's.
    pub fn broadcast(&self, root: u32, data: &mut [f32]) -> Result<(), CollectiveError> {
        let n = self.world_size();
        if n <= 1 {
            return Ok(());
        }
        if self.rank() == root {
            for r in 0..n {
                if r != root {
                    self.send_f32_tagged(r, TAG_BROADCAST, data)?;
                }
            }
        } else {
            let res = self.recv_f32_tagged(root, TAG_BROADCAST)?;
            if res.len() != data.len() {
                return Err(CollectiveError::LengthMismatch {
                    expected: data.len(),
                    got: res.len(),
                });
            }
            data.copy_from_slice(&res);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Condvar, Mutex};

    /// In-process [`Transport`] over shared queues — exercises the
    /// `ProcessGroup` collectives without sockets. One shared mailbox
    /// keyed by `(dst, src, tag)`; `send` pushes, `recv` blocks-pops.
    struct ChannelTransport {
        rank: u32,
        world: u32,
        mailbox: Arc<(Mutex<HashMap<(u32, u32, u32), VecDeque<Vec<u8>>>>, Condvar)>,
    }

    impl ChannelTransport {
        fn fan_out(world: u32) -> Vec<Arc<ChannelTransport>> {
            let mailbox = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));
            (0..world)
                .map(|r| {
                    Arc::new(ChannelTransport {
                        rank: r,
                        world,
                        mailbox: mailbox.clone(),
                    })
                })
                .collect()
        }
    }

    impl Transport for ChannelTransport {
        fn rank(&self) -> u32 {
            self.rank
        }
        fn world_size(&self) -> u32 {
            self.world
        }
        fn send_bytes(&self, to: u32, tag: u32, bytes: &[u8]) -> Result<(), CollectiveError> {
            let (m, cv) = &*self.mailbox;
            m.lock()
                .unwrap()
                .entry((to, self.rank, tag))
                .or_default()
                .push_back(bytes.to_vec());
            cv.notify_all();
            Ok(())
        }
        fn recv_bytes(&self, from: u32, tag: u32) -> Result<Vec<u8>, CollectiveError> {
            let (m, cv) = &*self.mailbox;
            let mut guard = m.lock().unwrap();
            loop {
                if let Some(q) = guard.get_mut(&(self.rank, from, tag))
                    && let Some(v) = q.pop_front()
                {
                    return Ok(v);
                }
                guard = cv.wait(guard).unwrap();
            }
        }
    }

    fn run_ranks<F>(world: u32, body: F) -> Vec<()>
    where
        F: Fn(ProcessGroup) + Send + Sync + 'static,
    {
        let ts = ChannelTransport::fan_out(world);
        let body = Arc::new(body);
        let handles: Vec<_> = ts
            .into_iter()
            .map(|t| {
                let body = body.clone();
                std::thread::spawn(move || body(ProcessGroup::new(t)))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    }

    #[test]
    fn all_reduce_sum_matches_serial() {
        run_ranks(4, |g| {
            let r = g.rank() as f32;
            let mut data = vec![r + 1.0; 3]; // ranks hold 1,2,3,4
            g.all_reduce(&mut data, ReduceKind::Sum).unwrap();
            assert_eq!(data, vec![10.0; 3], "rank {}", g.rank());
        });
    }

    #[test]
    fn all_gather_concatenates_in_rank_order() {
        run_ranks(3, |g| {
            let r = g.rank() as f32;
            let out = g.all_gather(&[10.0 * r, 10.0 * r + 1.0]).unwrap();
            assert_eq!(
                out,
                vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
                "rank {}",
                g.rank()
            );
        });
    }

    #[test]
    fn broadcast_from_root_overwrites() {
        run_ranks(4, |g| {
            let mut data = if g.is_leader() {
                vec![7.0, 8.0, 9.0]
            } else {
                vec![0.0, 0.0, 0.0]
            };
            g.broadcast(0, &mut data).unwrap();
            assert_eq!(data, vec![7.0, 8.0, 9.0], "rank {}", g.rank());
        });
    }

    #[test]
    fn barrier_round_trips() {
        run_ranks(4, |g| {
            g.barrier().unwrap();
            g.barrier().unwrap(); // repeated barriers stay in lockstep
        });
    }

    #[test]
    fn point_to_point_ring_handoff() {
        // Pipeline-style: each rank sends its id to the next, lowest
        // rank closes the ring. Verifies tagged matched send/recv.
        run_ranks(3, |g| {
            let n = g.world_size();
            let me = g.rank();
            let next = (me + 1) % n;
            let prev = (me + n - 1) % n;
            // Stagger to avoid all-send-then-all-recv lockstep issues:
            // even ranks send first, odd ranks recv first.
            if me % 2 == 0 {
                g.send_f32(next, 1, &[me as f32]).unwrap();
                let got = g.recv_f32(prev, 1).unwrap();
                assert_eq!(got, vec![prev as f32]);
            } else {
                let got = g.recv_f32(prev, 1).unwrap();
                assert_eq!(got, vec![prev as f32]);
                g.send_f32(next, 1, &[me as f32]).unwrap();
            }
        });
    }
}
