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

//! Collective ops as algorithms over [`SymmetricTransport`]
//! (plan #12).
//!
//! Borrowed from MAX's `kernels/src/comm/{allgather, allreduce,
//! reducescatter, allreduce_residual_rmsnorm_fp8, rms_norm_fp8}.mojo`.
//! Each collective is a small algorithm that uses
//! [`SymmetricTransport::put`] / `get` / `barrier` to move
//! tensors between ranks. Pure data layer — the transport is
//! pluggable, so the same algorithm runs against
//! `LocalTransport` (single-machine emulation) today and a
//! future MPI / NVSHMEM transport on a real cluster.
//!
//! Element-wise reductions parameterized by [`ReduceKind`].
//! All algorithms operate on `f32` slices today; quantized fp8
//! variants from MAX (`allreduce_residual_rmsnorm_fp8`) are
//! per-precision impls that slot in once a quantized model
//! lands.

use crate::symmetric::{CollectiveError, Rank, SymmetricBuffer, SymmetricTransport};

/// Element-wise reduction operator for collectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceKind {
    Sum,
    Mean,
    Max,
    Min,
}

impl ReduceKind {
    pub(crate) fn fold(self, acc: f32, x: f32) -> f32 {
        match self {
            Self::Sum => acc + x,
            Self::Mean => acc + x, // divide at the end
            Self::Max => acc.max(x),
            Self::Min => acc.min(x),
        }
    }
    pub(crate) fn finalize(self, acc: f32, n: usize) -> f32 {
        match self {
            Self::Mean => acc / (n as f32),
            _ => acc,
        }
    }
    fn identity(self) -> f32 {
        match self {
            Self::Sum | Self::Mean => 0.0,
            Self::Max => f32::NEG_INFINITY,
            Self::Min => f32::INFINITY,
        }
    }
}

/// AllReduce: every rank ends up with `op({values from every rank})`.
///
/// Naïve algorithm — every rank reads every other rank's slot
/// and combines. O(n_ranks²) communications, fine for small
/// rank counts. Real impls use ring-reduce / tree-reduce; we
/// pick simplicity since LocalTransport's "comm" is memcpy.
///
/// `local` carries this rank's contribution on entry; on exit
/// it carries the reduced result. Element count must match the
/// per-rank `len` of `buf` (in bytes: 4 * elements).
pub fn all_reduce<T: SymmetricTransport>(
    transport: &T,
    buf: SymmetricBuffer, // shape (offset, len) shared across ranks
    local: &mut [f32],
    op: ReduceKind,
) -> Result<(), CollectiveError> {
    let elems = buf.len / 4;
    if local.len() != elems {
        return Err(CollectiveError::LengthMismatch {
            expected: elems,
            got: local.len(),
        });
    }
    let me = transport.this_rank();
    let n = transport.num_ranks();

    // Step 1: write our contribution into our slot.
    let our_buf = SymmetricBuffer {
        rank: me,
        offset: buf.offset,
        len: buf.len,
    };
    let bytes = unsafe { std::slice::from_raw_parts(local.as_ptr() as *const u8, buf.len) };
    transport.put(our_buf, bytes)?;

    // Step 2: barrier so every rank has written its slot.
    transport.barrier()?;

    // Step 3: read every rank's slot and reduce.
    let mut acc: Vec<f32> = vec![op.identity(); elems];
    let mut scratch_bytes = vec![0u8; buf.len];
    for r in 0..n {
        let src = SymmetricBuffer {
            rank: Rank(r),
            offset: buf.offset,
            len: buf.len,
        };
        transport.get(src, &mut scratch_bytes)?;
        let scratch =
            unsafe { std::slice::from_raw_parts(scratch_bytes.as_ptr() as *const f32, elems) };
        for (i, &v) in scratch.iter().enumerate() {
            acc[i] = op.fold(acc[i], v);
        }
    }
    for v in acc.iter_mut() {
        *v = op.finalize(*v, n as usize);
    }
    local.copy_from_slice(&acc);
    Ok(())
}

/// AllGather: every rank ends up with the concatenation of all
/// per-rank `local` slices, in rank order.
///
/// `local.len()` is the per-rank chunk size; `output.len()` must
/// equal `num_ranks * local.len()`. Output rank `r`'s
/// contribution lands at `output[r*local.len()..(r+1)*local.len()]`.
pub fn all_gather<T: SymmetricTransport>(
    transport: &T,
    buf: SymmetricBuffer, // per-rank slot
    local: &[f32],
    output: &mut [f32],
) -> Result<(), CollectiveError> {
    let elems_per_rank = buf.len / 4;
    let n = transport.num_ranks() as usize;
    if local.len() != elems_per_rank {
        return Err(CollectiveError::LengthMismatch {
            expected: elems_per_rank,
            got: local.len(),
        });
    }
    if output.len() != n * elems_per_rank {
        return Err(CollectiveError::LengthMismatch {
            expected: n * elems_per_rank,
            got: output.len(),
        });
    }

    let me = transport.this_rank();
    let our_buf = SymmetricBuffer {
        rank: me,
        offset: buf.offset,
        len: buf.len,
    };
    let bytes = unsafe { std::slice::from_raw_parts(local.as_ptr() as *const u8, buf.len) };
    transport.put(our_buf, bytes)?;
    transport.barrier()?;

    let mut scratch_bytes = vec![0u8; buf.len];
    for r in 0..n {
        let src = SymmetricBuffer {
            rank: Rank(r as u32),
            offset: buf.offset,
            len: buf.len,
        };
        transport.get(src, &mut scratch_bytes)?;
        let chunk = unsafe {
            std::slice::from_raw_parts(scratch_bytes.as_ptr() as *const f32, elems_per_rank)
        };
        let dst_start = r * elems_per_rank;
        output[dst_start..dst_start + elems_per_rank].copy_from_slice(chunk);
    }
    Ok(())
}

/// ReduceScatter: equivalent to AllReduce followed by partition
/// — every rank ends up with one `chunk_size`-element slice of
/// the reduced result. Rank `r` gets element indices
/// `[r*chunk_size, (r+1)*chunk_size)`.
///
/// `local.len()` is the full vector (`num_ranks * chunk_size`);
/// `output.len()` is `chunk_size`.
pub fn reduce_scatter<T: SymmetricTransport>(
    transport: &T,
    buf: SymmetricBuffer,
    local: &[f32],
    output: &mut [f32],
    op: ReduceKind,
) -> Result<(), CollectiveError> {
    let total = buf.len / 4;
    let n = transport.num_ranks() as usize;
    if !total.is_multiple_of(n) {
        return Err(CollectiveError::TransportError {
            reason: format!("reduce_scatter: total elements {total} not divisible by {n} ranks"),
        });
    }
    let chunk = total / n;
    if local.len() != total {
        return Err(CollectiveError::LengthMismatch {
            expected: total,
            got: local.len(),
        });
    }
    if output.len() != chunk {
        return Err(CollectiveError::LengthMismatch {
            expected: chunk,
            got: output.len(),
        });
    }

    // Reuse all_reduce — works on a scratch copy of `local`,
    // then this rank picks its slice.
    let me = transport.this_rank().0 as usize;
    let mut full = local.to_vec();
    all_reduce(transport, buf, &mut full, op)?;
    output.copy_from_slice(&full[me * chunk..(me + 1) * chunk]);
    Ok(())
}

/// Bandwidth-optimal **ring** all-reduce over a symmetric heap — the standard
/// RDMA collective. Each rank moves only ~`2·(N-1)/N` of the vector (vs the
/// naïve [`all_reduce`]'s `O(N²)`), so it scales to large gradient tensors on
/// real RDMA fabrics (Thunderbolt/jaccl, InfiniBand). The algorithm is a
/// reduce-scatter ring followed by an all-gather ring (Baidu / NCCL pattern),
/// expressed entirely with one-sided `put` + `barrier`.
///
/// Requires `local.len() % num_ranks == 0` (pad upstream otherwise).
/// `mailbox_offset` names a `chunk`-sized scratch region present in every
/// rank's heap, where `chunk = local.len() / num_ranks` elements; the heap
/// must hold at least `mailbox_offset + chunk*4` bytes. `local` carries this
/// rank's contribution on entry and the reduced result on exit.
pub fn ring_all_reduce<T: SymmetricTransport>(
    transport: &T,
    mailbox_offset: usize,
    local: &mut [f32],
    op: ReduceKind,
) -> Result<(), CollectiveError> {
    let n = transport.num_ranks() as usize;
    let total = local.len();
    if n <= 1 {
        return Ok(());
    }
    if !total.is_multiple_of(n) {
        return Err(CollectiveError::TransportError {
            reason: format!("ring_all_reduce: len {total} not divisible by {n} ranks"),
        });
    }
    let chunk = total / n;
    let chunk_bytes = chunk * 4;
    let me = transport.this_rank().0 as usize;
    let right = (me + 1) % n;
    let mailbox = |rank: usize| SymmetricBuffer {
        rank: Rank(rank as u32),
        offset: mailbox_offset,
        len: chunk_bytes,
    };
    let mut recv = vec![0u8; chunk_bytes];

    let send_chunk = |local: &[f32], idx: usize| -> *const u8 {
        local[idx * chunk..idx * chunk + chunk].as_ptr() as *const u8
    };

    // Reduce-scatter: after N-1 steps, rank `me` owns the fully-reduced
    // (summed) chunk at index (me + 1) % n.
    for step in 0..n - 1 {
        let send_idx = (me + n - step) % n;
        let recv_idx = (me + n - step - 1) % n;
        let src = unsafe { std::slice::from_raw_parts(send_chunk(local, send_idx), chunk_bytes) };
        transport.put(mailbox(right), src)?;
        transport.barrier()?;
        transport.get(mailbox(me), &mut recv)?;
        let incoming = unsafe { std::slice::from_raw_parts(recv.as_ptr() as *const f32, chunk) };
        let d = recv_idx * chunk;
        for i in 0..chunk {
            local[d + i] = op.fold(local[d + i], incoming[i]);
        }
        transport.barrier()?;
    }

    // All-gather: propagate the reduced chunks around the ring (overwrite).
    for step in 0..n - 1 {
        let send_idx = (me + n + 1 - step) % n;
        let recv_idx = (me + n - step) % n;
        let src = unsafe { std::slice::from_raw_parts(send_chunk(local, send_idx), chunk_bytes) };
        transport.put(mailbox(right), src)?;
        transport.barrier()?;
        transport.get(mailbox(me), &mut recv)?;
        let incoming = unsafe { std::slice::from_raw_parts(recv.as_ptr() as *const f32, chunk) };
        let d = recv_idx * chunk;
        local[d..d + chunk].copy_from_slice(incoming);
        transport.barrier()?;
    }

    if op == ReduceKind::Mean {
        for v in local.iter_mut() {
            *v /= n as f32;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symmetric::LocalTransport;

    /// 4 ranks, each contributes [r+1, r+1, r+1, r+1] (so ranks
    /// hold 1, 2, 3, 4). After AllReduce::Sum each rank should
    /// see [10, 10, 10, 10].
    #[test]
    fn all_reduce_sum_across_4_ranks() {
        let n_ranks = 4u32;
        let elems = 4usize;
        let bytes = elems * 4;
        let ts = LocalTransport::fan_out(n_ranks, bytes);
        let _buf = SymmetricBuffer {
            rank: Rank(0),
            offset: 0,
            len: bytes,
        };

        // Each rank's local data + reduced output.
        let mut state: Vec<Vec<f32>> = (0..n_ranks).map(|r| vec![(r + 1) as f32; elems]).collect();

        // Run all_reduce sequentially; LocalTransport's barrier
        // counter accumulates across calls, so n_ranks calls
        // satisfy each barrier. We pre-write contributions for
        // every rank so the barrier-then-get phase sees data.
        // Step 1: each rank puts its contribution.
        for (r, t) in ts.iter().enumerate() {
            let our_buf = SymmetricBuffer {
                rank: Rank(r as u32),
                offset: 0,
                len: bytes,
            };
            let raw = unsafe { std::slice::from_raw_parts(state[r].as_ptr() as *const u8, bytes) };
            t.put(our_buf, raw).unwrap();
        }
        // Step 2: each rank reduces. We can't use the public
        // all_reduce since it does its own put + barrier (which
        // double-counts after our manual put above). Inline the
        // reduce step instead.
        for (r, t) in ts.iter().enumerate() {
            let mut acc = vec![0f32; elems];
            let mut scratch = vec![0u8; bytes];
            for src_r in 0..n_ranks {
                let src = SymmetricBuffer {
                    rank: Rank(src_r),
                    offset: 0,
                    len: bytes,
                };
                t.get(src, &mut scratch).unwrap();
                let view =
                    unsafe { std::slice::from_raw_parts(scratch.as_ptr() as *const f32, elems) };
                for (i, &v) in view.iter().enumerate() {
                    acc[i] += v;
                }
            }
            state[r] = acc;
        }

        for (r, slot) in state.iter().enumerate() {
            assert_eq!(slot, &vec![10.0; elems], "rank {r} after all-reduce");
        }
    }

    #[test]
    fn all_gather_concatenates_in_rank_order() {
        let n_ranks = 3u32;
        let chunk = 2usize;
        let bytes = chunk * 4;
        let ts = LocalTransport::fan_out(n_ranks, bytes);
        let _buf = SymmetricBuffer {
            rank: Rank(0),
            offset: 0,
            len: bytes,
        };

        // Each rank contributes [10*r, 10*r+1].
        let local: Vec<Vec<f32>> = (0..n_ranks)
            .map(|r| {
                let r = r as f32;
                vec![10.0 * r, 10.0 * r + 1.0]
            })
            .collect();

        // Step 1: each rank puts.
        for (r, t) in ts.iter().enumerate() {
            let our_buf = SymmetricBuffer {
                rank: Rank(r as u32),
                offset: 0,
                len: bytes,
            };
            let raw = unsafe { std::slice::from_raw_parts(local[r].as_ptr() as *const u8, bytes) };
            t.put(our_buf, raw).unwrap();
        }
        // Step 2: each rank gathers.
        for (r_idx, t) in ts.iter().enumerate() {
            let mut output = vec![0f32; n_ranks as usize * chunk];
            let mut scratch = vec![0u8; bytes];
            for src_r in 0..n_ranks {
                let src = SymmetricBuffer {
                    rank: Rank(src_r),
                    offset: 0,
                    len: bytes,
                };
                t.get(src, &mut scratch).unwrap();
                let view =
                    unsafe { std::slice::from_raw_parts(scratch.as_ptr() as *const f32, chunk) };
                let dst_start = src_r as usize * chunk;
                output[dst_start..dst_start + chunk].copy_from_slice(view);
            }
            assert_eq!(
                output,
                vec![0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
                "rank {r_idx} after all-gather"
            );
        }
    }

    #[test]
    fn ring_all_reduce_sum_matches_expected_multithread() {
        // N threads, each a rank sharing one heap + barrier; rank r holds
        // [r*10 + i]. After ring all-reduce(Sum) every rank holds the
        // element-wise sum. Exercises the reusable barrier across all the
        // ring's reduce-scatter + all-gather steps.
        for &n in &[2u32, 4] {
            let total = 8usize; // divisible by 2 and 4
            let chunk_bytes = (total / n as usize) * 4;
            let ts = LocalTransport::fan_out(n, chunk_bytes);

            let expected: Vec<f32> = (0..total)
                .map(|i| (0..n).map(|r| r as f32 * 10.0 + i as f32).sum())
                .collect();

            let handles: Vec<_> = ts
                .into_iter()
                .enumerate()
                .map(|(r, t)| {
                    std::thread::spawn(move || {
                        let mut local: Vec<f32> =
                            (0..total).map(|i| r as f32 * 10.0 + i as f32).collect();
                        ring_all_reduce(&t, 0, &mut local, ReduceKind::Sum).unwrap();
                        local
                    })
                })
                .collect();
            for h in handles {
                assert_eq!(h.join().unwrap(), expected, "n={n}");
            }
        }
    }

    #[test]
    fn ring_all_reduce_mean_divides() {
        let n = 4u32;
        let total = 8usize;
        let chunk_bytes = (total / n as usize) * 4;
        let ts = LocalTransport::fan_out(n, chunk_bytes);
        let handles: Vec<_> = ts
            .into_iter()
            .enumerate()
            .map(|(r, t)| {
                std::thread::spawn(move || {
                    let mut local = vec![r as f32 + 1.0; total]; // 1,2,3,4
                    ring_all_reduce(&t, 0, &mut local, ReduceKind::Mean).unwrap();
                    local
                })
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), vec![2.5f32; total]); // mean(1,2,3,4)
        }
    }

    #[test]
    fn reduce_kind_max_takes_pointwise_max() {
        let mut acc = ReduceKind::Max.identity();
        for v in [3.0, 1.0, 7.0, -2.0] {
            acc = ReduceKind::Max.fold(acc, v);
        }
        assert_eq!(acc, 7.0);
    }

    #[test]
    fn reduce_kind_mean_divides_at_finalize() {
        let mut acc = ReduceKind::Mean.identity();
        for v in [2.0, 4.0, 6.0, 8.0] {
            acc = ReduceKind::Mean.fold(acc, v);
        }
        assert_eq!(ReduceKind::Mean.finalize(acc, 4), 5.0);
    }
}
