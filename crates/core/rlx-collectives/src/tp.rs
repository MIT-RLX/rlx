// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Tensor parallelism: a group handle, weight-shard math, and HIR builders.
//!
//! Megatron-style tensor parallelism splits each layer's weights across the
//! ranks of a group so that every rank computes a *partial* result, then sums
//! the partials with one all-reduce:
//!
//! * a **column-parallel** projection `[in, out]` keeps the columns
//!   (output features) in [`TpGroup::shard`]`(out)` — attention heads, FFN
//!   intermediate channels. Its output is this rank's slice, no collective.
//! * a **row-parallel** projection `[in, out]` keeps the rows in
//!   `shard(in)` and reads the column-parallel slice; its output is a
//!   partial sum over ranks, finished by [`HirTpExt::tp_all_reduce`].
//!
//! A layer built that way is the unsharded layer with the head / channel
//! counts divided by `world`, plus one `tp_all_reduce` after each row-parallel
//! projection. With [`TpGroup::single`] every collective is the identity, so
//! one builder serves sharded and unsharded runs.
//!
//! ```
//! use rlx_collectives::tp::{TpGroup, run_local_ranks, shard_cols, shard_rows};
//!
//! // y = x @ A @ B with A column-parallel and B row-parallel, on 2 ranks.
//! let (n, k, m) = (1usize, 4usize, 4usize);
//! let x = vec![1.0f32; n * k];
//! let a: Vec<f32> = (0..k * m).map(|i| i as f32 * 0.1).collect();
//! let b: Vec<f32> = (0..m * k).map(|i| 1.0 - i as f32 * 0.05).collect();
//! let parts = run_local_ranks(2, move |tp: TpGroup| {
//!     let cols = tp.shard(m).unwrap();
//!     let a_r = shard_cols(&a, k, m, cols.clone()); // [k, m/2]
//!     let b_r = shard_rows(&b, m, k, cols); // [m/2, k]
//!     // ... build a graph with these, finish with `tp_all_reduce` ...
//!     (a_r.len(), b_r.len())
//! });
//! assert_eq!(parts, vec![(8, 8), (8, 8)]);
//! ```

use crate::{ALL_GATHER, ALL_REDUCE, REDUCE_SCATTER, register, register_group, unregister_group};
use rlx_driver::{InProcessTransport, ProcessGroup, TcpTransport, Transport};
use rlx_ir::hir::{HirMut, HirNodeId};
use rlx_ir::{HirGraphExt, Op, Shape};
use std::fmt;
use std::net::SocketAddr;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Group ids handed out by [`TpGroup::register`]. Starts high so it never
/// collides with the small hand-picked ids existing callers use.
static NEXT_GROUP_ID: AtomicU64 = AtomicU64::new(1 << 40);

/// A shard that does not divide evenly, or a rank outside its group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpError {
    /// `extent` (heads, channels, …) is not a multiple of the group size.
    Uneven { extent: usize, world: u32 },
    /// A block list whose sizes do not match the weight's width.
    BlockMismatch { blocks: usize, cols: usize },
}

impl fmt::Display for TpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TpError::Uneven { extent, world } => write!(
                f,
                "tensor parallel: extent {extent} does not split evenly over {world} ranks"
            ),
            TpError::BlockMismatch { blocks, cols } => write!(
                f,
                "tensor parallel: column blocks sum to {blocks}, weight has {cols} columns"
            ),
        }
    }
}

impl std::error::Error for TpError {}

/// One rank's view of a tensor-parallel group: its rank, the group size and
/// the id its [`ProcessGroup`] is registered under. `Copy`, so it threads
/// through graph builders like a dims struct.
///
/// Build one with [`TpGroup::single`] (no parallelism), [`TpGroup::register`]
/// (any transport), [`TpGroup::tcp`] (one process per rank over TCP), or let
/// [`run_local_ranks`] create one per thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TpGroup {
    rank: u32,
    world: u32,
    group_id: u64,
}

impl Default for TpGroup {
    fn default() -> Self {
        Self::single()
    }
}

impl TpGroup {
    /// The trivial group: rank 0 of 1. Every collective is the identity and
    /// every shard is the whole extent.
    pub const fn single() -> Self {
        Self {
            rank: 0,
            world: 1,
            group_id: 0,
        }
    }

    /// Register `group` under a fresh id (so several groups — tensor and
    /// pipeline, or several in-process ranks — never share one) and return
    /// this rank's handle. Also installs the collective ops ([`register`]).
    /// Call [`TpGroup::release`] when done to drop the registry's reference.
    pub fn register(group: Arc<ProcessGroup>) -> Self {
        register();
        let (rank, world) = (group.rank(), group.world_size());
        if world == 1 {
            return Self::single();
        }
        let group_id = NEXT_GROUP_ID.fetch_add(1, Ordering::Relaxed);
        register_group(group_id, group);
        Self {
            rank,
            world,
            group_id,
        }
    }

    /// [`TpGroup::register`] over any transport.
    pub fn with_transport(transport: Arc<dyn Transport>) -> Self {
        Self::register(Arc::new(ProcessGroup::new(transport)))
    }

    /// One process per rank over a full TCP mesh. `peers[r]` is rank `r`'s
    /// address; every rank passes the same list.
    pub fn tcp(rank: u32, peers: Vec<SocketAddr>) -> std::io::Result<Self> {
        let world = peers.len() as u32;
        if world <= 1 {
            return Ok(Self::single());
        }
        let t = TcpTransport::bind(rank, world, peers, 1 << 20)?;
        Ok(Self::with_transport(Arc::new(t)))
    }

    /// Drop the registry's reference to this group's [`ProcessGroup`].
    /// Graphs built with this handle must not run afterwards.
    pub fn release(self) {
        if !self.is_single() {
            unregister_group(self.group_id);
        }
    }

    pub fn rank(&self) -> u32 {
        self.rank
    }

    pub fn world(&self) -> u32 {
        self.world
    }

    /// The id the group is registered under (what collective `attrs` carry).
    pub fn group_id(&self) -> u64 {
        self.group_id
    }

    /// The registered [`ProcessGroup`], for collectives on host buffers
    /// (`None` for [`TpGroup::single`] or after [`TpGroup::release`]).
    pub fn process_group(&self) -> Option<Arc<ProcessGroup>> {
        if self.is_single() {
            None
        } else {
            crate::lookup_group(self.group_id)
        }
    }

    /// Sum a host buffer across the group in place (no-op for a group of one).
    pub fn all_reduce_host(&self, data: &mut [f32]) -> Result<(), rlx_driver::CollectiveError> {
        match self.process_group() {
            None if self.is_single() => Ok(()),
            None => Err(rlx_driver::CollectiveError::TransportError {
                reason: format!("tensor-parallel group {} is not registered", self.group_id),
            }),
            Some(g) => g.all_reduce(data, rlx_driver::ReduceKind::Sum),
        }
    }

    /// Overwrite every rank's `data` with rank `root`'s (no-op for a group of
    /// one). For decisions every rank must take identically — routing, sampled
    /// tokens — where ranks on different devices could otherwise round apart.
    pub fn broadcast_host(
        &self,
        root: u32,
        data: &mut [f32],
    ) -> Result<(), rlx_driver::CollectiveError> {
        match self.process_group() {
            None if self.is_single() => Ok(()),
            None => Err(rlx_driver::CollectiveError::TransportError {
                reason: format!("tensor-parallel group {} is not registered", self.group_id),
            }),
            Some(g) => g.broadcast(root, data),
        }
    }

    /// `true` for a group of one: no collectives are emitted.
    pub fn is_single(&self) -> bool {
        self.world == 1
    }

    /// This rank's contiguous slice of `extent` items split evenly over the
    /// group (`extent / world` each).
    pub fn shard(&self, extent: usize) -> Result<Range<usize>, TpError> {
        let n = self.local(extent)?;
        let start = self.rank as usize * n;
        Ok(start..start + n)
    }

    /// `extent / world`, or an error when it does not divide evenly.
    pub fn local(&self, extent: usize) -> Result<usize, TpError> {
        let w = self.world as usize;
        if !extent.is_multiple_of(w) {
            return Err(TpError::Uneven {
                extent,
                world: self.world,
            });
        }
        Ok(extent / w)
    }

    /// Like [`TpGroup::shard`] over whole units of `unit` items each (e.g.
    /// heads of `head_dim` channels): the item range of this rank's units.
    pub fn shard_units(&self, units: usize, unit: usize) -> Result<Range<usize>, TpError> {
        let r = self.shard(units)?;
        Ok(r.start * unit..r.end * unit)
    }
}

// ── host-side weight slicing ──────────────────────────────────────

/// Columns `cols` of a row-major `[rows, width]` matrix → `[rows, cols.len()]`.
/// The column-parallel slice of an `[in, out]` projection.
pub fn shard_cols<T: Copy>(w: &[T], rows: usize, width: usize, cols: Range<usize>) -> Vec<T> {
    assert_eq!(
        w.len(),
        rows * width,
        "shard_cols: weight is not [rows, width]"
    );
    assert!(cols.end <= width, "shard_cols: columns out of range");
    let mut out = Vec::with_capacity(rows * cols.len());
    for r in 0..rows {
        out.extend_from_slice(&w[r * width + cols.start..r * width + cols.end]);
    }
    out
}

/// Rows `rows_r` of a row-major `[rows, width]` matrix → `[rows_r.len(), width]`.
/// The row-parallel slice of an `[in, out]` projection, and the slice of any
/// per-channel vector (`width = 1`).
pub fn shard_rows<T: Copy>(w: &[T], rows: usize, width: usize, rows_r: Range<usize>) -> Vec<T> {
    assert_eq!(
        w.len(),
        rows * width,
        "shard_rows: weight is not [rows, width]"
    );
    assert!(rows_r.end <= rows, "shard_rows: rows out of range");
    w[rows_r.start * width..rows_r.end * width].to_vec()
}

/// Column-parallel slice of a matrix whose columns are several **blocks laid
/// side by side** (a fused `[gate | up]` projection, `[q | k | v]`): each block
/// is split over the group on its own, and this rank's pieces are
/// concatenated in block order. `blocks` are the block widths.
pub fn shard_col_blocks<T: Copy>(
    w: &[T],
    rows: usize,
    blocks: &[usize],
    tp: &TpGroup,
) -> Result<Vec<T>, TpError> {
    let width: usize = blocks.iter().sum();
    if w.len() != rows * width {
        return Err(TpError::BlockMismatch {
            blocks: width,
            cols: w.len() / rows.max(1),
        });
    }
    let ranges = blocks
        .iter()
        .scan(0usize, |off, &b| {
            let start = *off;
            *off += b;
            Some(tp.shard(b).map(|r| start + r.start..start + r.end))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let local: usize = ranges.iter().map(|r| r.len()).sum();
    let mut out = Vec::with_capacity(rows * local);
    for r in 0..rows {
        for c in &ranges {
            out.extend_from_slice(&w[r * width + c.start..r * width + c.end]);
        }
    }
    Ok(out)
}

/// [`shard_cols`] applied to each matrix of a `[batch, rows, width]` stack
/// (an expert bank).
pub fn shard_cols_batched<T: Copy>(
    w: &[T],
    batch: usize,
    rows: usize,
    width: usize,
    cols: Range<usize>,
) -> Vec<T> {
    assert_eq!(w.len(), batch * rows * width, "shard_cols_batched: shape");
    let m = rows * width;
    (0..batch)
        .flat_map(|e| shard_cols(&w[e * m..(e + 1) * m], rows, width, cols.clone()))
        .collect()
}

/// [`shard_col_blocks`] applied to each matrix of a `[batch, rows, Σblocks]` stack.
pub fn shard_col_blocks_batched<T: Copy>(
    w: &[T],
    batch: usize,
    rows: usize,
    blocks: &[usize],
    tp: &TpGroup,
) -> Result<Vec<T>, TpError> {
    let m = rows * blocks.iter().sum::<usize>();
    if w.len() != batch * m {
        return Err(TpError::BlockMismatch {
            blocks: m,
            cols: w.len() / batch.max(1),
        });
    }
    let mut out = Vec::new();
    for e in 0..batch {
        out.extend(shard_col_blocks(&w[e * m..(e + 1) * m], rows, blocks, tp)?);
    }
    Ok(out)
}

/// [`shard_rows`] applied to each matrix of a `[batch, rows, width]` stack.
pub fn shard_rows_batched<T: Copy>(
    w: &[T],
    batch: usize,
    rows: usize,
    width: usize,
    rows_r: Range<usize>,
) -> Vec<T> {
    assert_eq!(w.len(), batch * rows * width, "shard_rows_batched: shape");
    let m = rows * width;
    (0..batch)
        .flat_map(|e| shard_rows(&w[e * m..(e + 1) * m], rows, width, rows_r.clone()))
        .collect()
}

// ── in-graph collectives on the HIR ───────────────────────────────

/// Collectives on a [`HirMut`] graph under a [`TpGroup`]. Each is the
/// identity for [`TpGroup::single`], so a builder can call them
/// unconditionally.
pub trait HirTpExt {
    /// Sum `x` across the group (shape preserved) — finishes a row-parallel
    /// projection.
    fn tp_all_reduce(&mut self, x: HirNodeId, tp: &TpGroup) -> HirNodeId;
    /// Concatenate each rank's `x` along axis 0 (extent × world).
    fn tp_all_gather(&mut self, x: HirNodeId, tp: &TpGroup) -> HirNodeId;
    /// Sum across the group, keep this rank's axis-0 slice (extent ÷ world).
    fn tp_reduce_scatter(&mut self, x: HirNodeId, tp: &TpGroup) -> HirNodeId;
    /// Concatenate each rank's `x` along `axis` (any axis): all-gather on a
    /// moved-to-front view. Gathers column-parallel outputs into full width.
    fn tp_all_gather_axis(&mut self, x: HirNodeId, axis: usize, tp: &TpGroup) -> HirNodeId;
}

fn custom(g: &mut HirMut, name: &str, attrs: Vec<u8>, x: HirNodeId, shape: Shape) -> HirNodeId {
    g.0.mir(
        Op::Custom {
            name: name.to_string(),
            num_inputs: 1,
            attrs,
        },
        vec![x],
        shape,
    )
}

fn group_ws_attrs(tp: &TpGroup) -> Vec<u8> {
    crate::encode_group_ws(tp.group_id, tp.world)
}

fn scaled_axis0(shape: &Shape, mul: usize, div: usize) -> Shape {
    let mut dims: Vec<usize> = shape.dims().iter().map(|d| d.unwrap_static()).collect();
    assert!(!dims.is_empty(), "collective on a rank-0 tensor");
    assert!(
        (dims[0] * mul).is_multiple_of(div),
        "reduce_scatter: axis 0 ({}) does not split over {div} ranks",
        dims[0]
    );
    dims[0] = dims[0] * mul / div;
    Shape::new(&dims, shape.dtype())
}

impl HirTpExt for HirMut<'_> {
    fn tp_all_reduce(&mut self, x: HirNodeId, tp: &TpGroup) -> HirNodeId {
        if tp.is_single() {
            return x;
        }
        register();
        let shape = self.shape(x).clone();
        let mut attrs = tp.group_id.to_le_bytes().to_vec();
        attrs.push(0); // ReduceKind::Sum
        custom(self, ALL_REDUCE, attrs, x, shape)
    }

    fn tp_all_gather(&mut self, x: HirNodeId, tp: &TpGroup) -> HirNodeId {
        if tp.is_single() {
            return x;
        }
        register();
        let shape = scaled_axis0(self.shape(x), tp.world as usize, 1);
        custom(self, ALL_GATHER, group_ws_attrs(tp), x, shape)
    }

    fn tp_reduce_scatter(&mut self, x: HirNodeId, tp: &TpGroup) -> HirNodeId {
        if tp.is_single() {
            return x;
        }
        register();
        let shape = scaled_axis0(self.shape(x), 1, tp.world as usize);
        custom(self, REDUCE_SCATTER, group_ws_attrs(tp), x, shape)
    }

    fn tp_all_gather_axis(&mut self, x: HirNodeId, axis: usize, tp: &TpGroup) -> HirNodeId {
        if tp.is_single() {
            return x;
        }
        let rank = self.shape(x).dims().len();
        assert!(
            axis < rank,
            "tp_all_gather_axis: axis {axis} of rank {rank}"
        );
        if axis == 0 {
            return self.tp_all_gather(x, tp);
        }
        // move `axis` to the front, gather, move it back
        let mut perm: Vec<usize> = (0..rank).collect();
        perm.remove(axis);
        perm.insert(0, axis);
        let front = self.transpose_(x, perm.clone());
        let gathered = self.tp_all_gather(front, tp);
        let mut inv = vec![0usize; rank];
        for (i, &p) in perm.iter().enumerate() {
            inv[p] = i;
        }
        self.transpose_(gathered, inv)
    }
}

// ── single-machine harness ────────────────────────────────────────

/// Run `body` once per rank of a `world`-rank group, each on its own thread
/// over an [`InProcessTransport`], and return the results in rank order.
/// Each rank gets its own [`TpGroup`] (released after `body` returns). For
/// testing a sharded program against its unsharded run on one machine.
///
/// Panics in any rank propagate (with the rank in the message).
pub fn run_local_ranks<T, F>(world: u32, body: F) -> Vec<T>
where
    T: Send + 'static,
    F: Fn(TpGroup) -> T + Send + Sync + 'static,
{
    assert!(world >= 1, "run_local_ranks: world must be >= 1");
    register();
    if world == 1 {
        return vec![body(TpGroup::single())];
    }
    let body = Arc::new(body);
    let handles: Vec<_> = InProcessTransport::fan_out(world)
        .into_iter()
        .map(|t| {
            let body = body.clone();
            std::thread::Builder::new()
                .name(format!("tp-rank-{}", t.rank()))
                .stack_size(64 << 20)
                .spawn(move || {
                    let tp = TpGroup::with_transport(t);
                    let out = body(tp);
                    tp.release();
                    out
                })
                .expect("spawn rank thread")
        })
        .collect();
    handles
        .into_iter()
        .enumerate()
        .map(|(r, h)| {
            h.join()
                .unwrap_or_else(|e| std::panic::resume_unwind(Box::new(format!("rank {r}: {e:?}"))))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_math() {
        let tp = TpGroup {
            rank: 1,
            world: 4,
            group_id: 0,
        };
        assert_eq!(tp.shard(8).unwrap(), 2..4);
        assert_eq!(tp.shard_units(4, 3).unwrap(), 3..6);
        assert!(matches!(tp.shard(6), Err(TpError::Uneven { .. })));
        assert_eq!(TpGroup::single().shard(7).unwrap(), 0..7);
    }

    #[test]
    fn slicing() {
        // [2, 4]
        let w = [0, 1, 2, 3, 4, 5, 6, 7];
        assert_eq!(shard_cols(&w, 2, 4, 1..3), vec![1, 2, 5, 6]);
        assert_eq!(shard_rows(&w, 2, 4, 1..2), vec![4, 5, 6, 7]);
        let r1 = TpGroup {
            rank: 1,
            world: 2,
            group_id: 0,
        };
        // blocks [gate:2 | up:2] → rank 1 keeps gate[1], up[1]
        assert_eq!(
            shard_col_blocks(&w, 2, &[2, 2], &r1).unwrap(),
            vec![1, 3, 5, 7]
        );
        let bank = [0, 1, 2, 3, 10, 11, 12, 13]; // [2, 1, 4]
        assert_eq!(shard_cols_batched(&bank, 2, 1, 4, 2..4), vec![2, 3, 12, 13]);
        assert_eq!(shard_rows_batched(&bank, 2, 2, 2, 1..2), vec![2, 3, 12, 13]);
    }

    #[test]
    fn local_ranks_all_reduce() {
        let sums = run_local_ranks(3, |tp| {
            let mut x = vec![tp.rank() as f32 + 1.0; 2];
            tp.all_reduce_host(&mut x).unwrap();
            x[0]
        });
        assert_eq!(sums, vec![6.0; 3]);
    }
}
