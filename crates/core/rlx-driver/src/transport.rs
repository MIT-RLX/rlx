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
use half::slice::HalfFloatSliceExt;
use rlx_ir::DType;
use std::sync::Arc;
use std::time::{Duration, Instant};

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

    /// Like [`Self::recv_bytes`] but gives up after `timeout`, returning
    /// `Ok(None)`. The default blocks (transports with no deadline support
    /// ignore the timeout); [`crate::NetTransport`] overrides it with a real
    /// `wait_timeout`. This is the basis for not letting a slow or offline
    /// edge node stall a collective (bounded staleness).
    fn recv_bytes_timeout(
        &self,
        from: u32,
        tag: u32,
        _timeout: Duration,
    ) -> Result<Option<Vec<u8>>, CollectiveError> {
        Ok(Some(self.recv_bytes(from, tag)?))
    }

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

/// f64 reduction step — collectives accumulate in f64 regardless of the
/// payload dtype, so f16/bf16 reductions don't lose range mid-sum.
fn combine64(op: ReduceKind, a: f64, b: f64) -> f64 {
    match op {
        ReduceKind::Sum | ReduceKind::Mean => a + b,
        ReduceKind::Max => a.max(b),
        ReduceKind::Min => a.min(b),
    }
}

/// The float dtypes a collective can reduce over the wire.
fn is_reducible_float(d: DType) -> bool {
    matches!(d, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
}

/// The integer dtypes a collective can reduce — the edge/quantized tier
/// (MCU/FPGA/ANE). Accumulated in i32 so a sum of 8-bit lanes can't overflow.
fn is_reducible_int(d: DType) -> bool {
    matches!(d, DType::I8 | DType::U8)
}

fn decode_to_i32(bytes: &[u8], dtype: DType) -> Vec<i32> {
    match dtype {
        DType::I8 => bytes.iter().map(|&b| b as i8 as i32).collect(),
        DType::U8 => bytes.iter().map(|&b| b as i32).collect(),
        _ => Vec::new(),
    }
}

fn encode_into_i32(dst: &mut [u8], vals: &[i32], dtype: DType) {
    match dtype {
        DType::I8 => {
            for (d, &v) in dst.iter_mut().zip(vals) {
                *d = v.clamp(i8::MIN as i32, i8::MAX as i32) as i8 as u8;
            }
        }
        DType::U8 => {
            for (d, &v) in dst.iter_mut().zip(vals) {
                *d = v.clamp(0, u8::MAX as i32) as u8;
            }
        }
        _ => {}
    }
}

fn combine_i32(op: ReduceKind, a: i32, b: i32) -> i32 {
    match op {
        ReduceKind::Sum | ReduceKind::Mean => a + b,
        ReduceKind::Max => a.max(b),
        ReduceKind::Min => a.min(b),
    }
}

/// Decode `bytes` of `dtype` to f64 for reduction. Little-endian on the wire.
fn decode_to_f64(bytes: &[u8], dtype: DType) -> Vec<f64> {
    match dtype {
        DType::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        DType::F64 => bytes
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        _ => Vec::new(),
    }
}

/// Encode f64 reduction results back to `dtype` bytes (little-endian).
fn encode_from_f64(vals: &[f64], dtype: DType) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * dtype.size_bytes().max(1));
    match dtype {
        DType::F32 => vals
            .iter()
            .for_each(|&v| out.extend_from_slice(&(v as f32).to_le_bytes())),
        DType::F64 => vals
            .iter()
            .for_each(|&v| out.extend_from_slice(&v.to_le_bytes())),
        DType::F16 => vals
            .iter()
            .for_each(|&v| out.extend_from_slice(&half::f16::from_f64(v).to_le_bytes())),
        DType::BF16 => vals
            .iter()
            .for_each(|&v| out.extend_from_slice(&half::bf16::from_f64(v).to_le_bytes())),
        _ => {}
    }
    out
}

/// f32 reduction step — the accumulator for f16/bf16/f32. f32 is exact for
/// f16/bf16 inputs and matches rlx's native f32 collective bit-for-bit.
fn combine32(op: ReduceKind, a: f32, b: f32) -> f32 {
    match op {
        ReduceKind::Sum | ReduceKind::Mean => a + b,
        ReduceKind::Max => a.max(b),
        ReduceKind::Min => a.min(b),
    }
}

/// SIMD-batch decode of `dtype` bytes to f32. On little-endian targets (Apple
/// Silicon + x86) f16/bf16 reinterpret in place and `half`'s vectorized
/// `convert_to_f32_slice` runs ~an order of magnitude faster than the
/// per-element scalar path; a scalar `from_le_bytes` fallback keeps big-endian
/// / misaligned slices correct.
fn decode_to_f32(bytes: &[u8], dtype: DType) -> Vec<f32> {
    match dtype {
        DType::F32 => {
            #[cfg(target_endian = "little")]
            {
                let (head, mid, tail) = unsafe { bytes.align_to::<f32>() };
                if head.is_empty() && tail.is_empty() {
                    return mid.to_vec();
                }
            }
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        DType::F16 => {
            let mut out = vec![0f32; bytes.len() / 2];
            #[cfg(target_endian = "little")]
            {
                // SAFETY: any 2-byte pattern is a valid f16 (no invalid bits);
                // LE in-memory layout equals the wire's LE f16.
                let (head, mid, tail) = unsafe { bytes.align_to::<half::f16>() };
                if head.is_empty() && tail.is_empty() {
                    mid.convert_to_f32_slice(&mut out);
                    return out;
                }
            }
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
            out
        }
        DType::BF16 => {
            let mut out = vec![0f32; bytes.len() / 2];
            #[cfg(target_endian = "little")]
            {
                let (head, mid, tail) = unsafe { bytes.align_to::<half::bf16>() };
                if head.is_empty() && tail.is_empty() {
                    mid.convert_to_f32_slice(&mut out);
                    return out;
                }
            }
            for (o, c) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f32();
            }
            out
        }
        _ => Vec::new(),
    }
}

/// SIMD-batch encode of f32 values **into** `dst` (raw `dtype` bytes), in
/// place. On little-endian, f16/bf16 convert straight into the reinterpreted
/// `dst` slice via `half`'s vectorized `convert_from_f32_slice` — no temporary
/// `Vec<f16>` and no `Vec<u8>`, the two allocations the old encode paid per
/// ring step. A scalar `to_le_bytes` path keeps BE / misaligned correct.
fn encode_into(dst: &mut [u8], vals: &[f32], dtype: DType) {
    match dtype {
        DType::F32 => {
            for (c, &v) in dst.chunks_exact_mut(4).zip(vals) {
                c.copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::F16 => {
            #[cfg(target_endian = "little")]
            {
                let (head, mid, tail) = unsafe { dst.align_to_mut::<half::f16>() };
                if head.is_empty() && tail.is_empty() {
                    mid.convert_from_f32_slice(vals);
                    return;
                }
            }
            for (c, &v) in dst.chunks_exact_mut(2).zip(vals) {
                c.copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        DType::BF16 => {
            #[cfg(target_endian = "little")]
            {
                let (head, mid, tail) = unsafe { dst.align_to_mut::<half::bf16>() };
                if head.is_empty() && tail.is_empty() {
                    mid.convert_from_f32_slice(vals);
                    return;
                }
            }
            for (c, &v) in dst.chunks_exact_mut(2).zip(vals) {
                c.copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        _ => {}
    }
}

/// Reduce `incoming` into `dst` (both raw `dtype` bytes), elementwise with
/// `op`. f16/bf16/f32 accumulate in f32 (SIMD codec); f64 stays in f64.
fn reduce_into(dst: &mut [u8], incoming: &[u8], dtype: DType, op: ReduceKind) {
    if is_reducible_int(dtype) {
        let mut acc = decode_to_i32(dst, dtype);
        let b = decode_to_i32(incoming, dtype);
        for (x, &y) in acc.iter_mut().zip(&b) {
            *x = combine_i32(op, *x, y);
        }
        encode_into_i32(dst, &acc, dtype);
    } else if dtype == DType::F64 {
        let a = decode_to_f64(dst, dtype);
        let b = decode_to_f64(incoming, dtype);
        let m: Vec<f64> = a
            .iter()
            .zip(&b)
            .map(|(&x, &y)| combine64(op, x, y))
            .collect();
        dst.copy_from_slice(&encode_from_f64(&m, dtype));
    } else {
        let mut acc = decode_to_f32(dst, dtype);
        let b = decode_to_f32(incoming, dtype);
        for (x, &y) in acc.iter_mut().zip(&b) {
            *x = combine32(op, *x, y);
        }
        encode_into(dst, &acc, dtype);
    }
}

/// Divide every element of `data` (raw `dtype` bytes) by `n` — the `Mean`
/// finalize, applied once over the whole reduced buffer.
fn scale_mean(data: &mut [u8], dtype: DType, n: usize) {
    if is_reducible_int(dtype) {
        // Integer mean (federated averaging): round-to-nearest i32 divide.
        let half = n as i32 / 2;
        let mut acc = decode_to_i32(data, dtype);
        for v in acc.iter_mut() {
            *v = (*v + if *v >= 0 { half } else { -half }) / n as i32;
        }
        encode_into_i32(data, &acc, dtype);
    } else if dtype == DType::F64 {
        let scaled: Vec<f64> = decode_to_f64(data, dtype)
            .iter()
            .map(|&v| v / n as f64)
            .collect();
        data.copy_from_slice(&encode_from_f64(&scaled, dtype));
    } else {
        let inv = 1.0 / n as f32;
        let mut acc = decode_to_f32(data, dtype);
        for v in acc.iter_mut() {
            *v *= inv;
        }
        encode_into(data, &acc, dtype);
    }
}

/// Pick a common reduce dtype for a **mixed-precision** mesh. Reduction
/// accumulates in f64 internally either way; this only decides the dtype the
/// ranks agree to exchange when their *local* dtypes differ. F64 if any rank
/// holds f64, else F32 — the safe accumulator (NCCL's default). Ranks that
/// share a dtype should just reduce in it natively (smaller wire).
pub fn negotiate_reduce_dtype(dtypes: &[DType]) -> DType {
    if dtypes.iter().any(|&d| d == DType::F64) {
        DType::F64
    } else {
        DType::F32
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
    /// Bandwidth-optimal **ring** all-reduce (reduce-scatter then
    /// all-gather): every rank moves only `~2·(n-1)/n · len` floats in and
    /// out, independent of `n`, versus the old gather-to-root that funneled
    /// `O(n)·len` through rank 0. Deadlock-free over [`NetTransport`]
    /// because each rank's background reader thread always drains its
    /// socket, so a `send` never blocks on a peer that is itself sending.
    pub fn all_reduce(&self, data: &mut [f32], op: ReduceKind) -> Result<(), CollectiveError> {
        let n = self.world_size();
        if n <= 1 {
            for v in data.iter_mut() {
                *v = finalize(op, *v, n.max(1));
            }
            return Ok(());
        }
        let n = n as usize;
        let me = self.rank() as usize;
        let next = ((me + 1) % n) as u32;
        let prev = ((me + n - 1) % n) as u32;

        // Near-equal contiguous chunks; the first `rem` get one extra elem.
        let len = data.len();
        let base = len / n;
        let rem = len % n;
        let bound = |i: usize| i * base + i.min(rem);
        let chunk = |i: usize| (bound(i), bound(i + 1));

        // Single tag is safe: the inbox is FIFO per source, and `prev`
        // sends its 2(n-1) frames in exactly the order this rank consumes
        // them.
        let expect = |incoming: &[f32], want: usize| -> Result<(), CollectiveError> {
            if incoming.len() != want {
                return Err(CollectiveError::LengthMismatch {
                    expected: want,
                    got: incoming.len(),
                });
            }
            Ok(())
        };

        // reduce-scatter: rank `me` ends owning the fully-reduced chunk (me+1)%n.
        for step in 0..n - 1 {
            let (ss, se) = chunk((me + n - step) % n);
            let (rs, re) = chunk((me + n - step - 1) % n);
            self.send_f32_tagged(next, TAG_ALL_REDUCE, &data[ss..se])?;
            let incoming = self.recv_f32_tagged(prev, TAG_ALL_REDUCE)?;
            expect(&incoming, re - rs)?;
            for (d, v) in data[rs..re].iter_mut().zip(incoming) {
                *d = combine(op, *d, v);
            }
        }
        // all-gather: walk each reduced chunk around the ring.
        for step in 0..n - 1 {
            let (ss, se) = chunk((me + n - step + 1) % n);
            let (rs, re) = chunk((me + n - step) % n);
            self.send_f32_tagged(next, TAG_ALL_REDUCE, &data[ss..se])?;
            let incoming = self.recv_f32_tagged(prev, TAG_ALL_REDUCE)?;
            expect(&incoming, re - rs)?;
            data[rs..re].copy_from_slice(&incoming);
        }
        // Mean is sum-reduced above, divided here once over the whole vector.
        if matches!(op, ReduceKind::Mean) {
            let inv = 1.0 / n as f32;
            for v in data.iter_mut() {
                *v *= inv;
            }
        }
        Ok(())
    }

    /// In-place all-reduce over **any float dtype** (`F16`/`BF16`/`F32`/`F64`),
    /// operating on the raw little-endian bytes. The wire carries the *native*
    /// dtype — an fp16 reduction moves half the bytes of the f32 path, the
    /// direct bandwidth win of mixed precision — while the reduction itself
    /// accumulates in f64 so range isn't lost mid-sum. Same bandwidth-optimal
    /// ring as [`Self::all_reduce`]. `data.len()` must be a multiple of the
    /// dtype's element size, and every rank must pass the same dtype + length.
    pub fn all_reduce_typed(
        &self,
        data: &mut [u8],
        dtype: DType,
        op: ReduceKind,
    ) -> Result<(), CollectiveError> {
        if !is_reducible_float(dtype) && !is_reducible_int(dtype) {
            return Err(CollectiveError::TransportError {
                reason: format!("all_reduce_typed: unsupported dtype {dtype:?}"),
            });
        }
        let esz = dtype.size_bytes();
        if esz == 0 || data.len() % esz != 0 {
            return Err(CollectiveError::TransportError {
                reason: format!(
                    "all_reduce_typed: {} bytes not a multiple of {esz}",
                    data.len()
                ),
            });
        }
        let n = self.world_size();
        if n <= 1 {
            return Ok(()); // Mean over 1 rank is identity
        }
        let n = n as usize;
        let me = self.rank() as usize;
        let next = ((me + 1) % n) as u32;
        let prev = ((me + n - 1) % n) as u32;

        let elems = data.len() / esz;
        let base = elems / n;
        let rem = elems % n;
        let ebound = |i: usize| (i * base + i.min(rem)) * esz; // byte offset of element-chunk i
        let chunk = |i: usize| (ebound(i), ebound(i + 1));

        // reduce-scatter (decode→combine in f64→encode back into our chunk)
        for step in 0..n - 1 {
            let (ss, se) = chunk((me + n - step) % n);
            let (rs, re) = chunk((me + n - step - 1) % n);
            self.transport
                .send_bytes(next, TAG_ALL_REDUCE, &data[ss..se])?;
            let incoming = self.transport.recv_bytes(prev, TAG_ALL_REDUCE)?;
            if incoming.len() != re - rs {
                return Err(CollectiveError::LengthMismatch {
                    expected: re - rs,
                    got: incoming.len(),
                });
            }
            reduce_into(&mut data[rs..re], &incoming, dtype, op);
        }
        // all-gather (raw byte copy — already reduced)
        for step in 0..n - 1 {
            let (ss, se) = chunk((me + n - step + 1) % n);
            let (rs, re) = chunk((me + n - step) % n);
            self.transport
                .send_bytes(next, TAG_ALL_REDUCE, &data[ss..se])?;
            let incoming = self.transport.recv_bytes(prev, TAG_ALL_REDUCE)?;
            if incoming.len() != re - rs {
                return Err(CollectiveError::LengthMismatch {
                    expected: re - rs,
                    got: incoming.len(),
                });
            }
            data[rs..re].copy_from_slice(&incoming);
        }
        if matches!(op, ReduceKind::Mean) {
            scale_mean(data, dtype, n);
        }
        Ok(())
    }

    /// Non-blocking all-reduce: takes ownership of `data`, runs the
    /// collective on a background thread, and returns a handle whose
    /// `join()` yields the reduced vector. This is the building block for
    /// overlapping gradient communication with backward compute — fire it
    /// per bucket as gradients are produced, keep computing, then join.
    pub fn spawn_all_reduce(
        self: &Arc<Self>,
        mut data: Vec<f32>,
        op: ReduceKind,
    ) -> std::thread::JoinHandle<Vec<f32>> {
        let g = self.clone();
        std::thread::spawn(move || {
            g.all_reduce(&mut data, op).expect("background all_reduce");
            data
        })
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

    /// Bounded-staleness federated averaging. Rank 0 collects each rank's
    /// vector but **skips any that miss `deadline`**, averages the arrivals
    /// (always including its own), and broadcasts the result. Returns the
    /// participant count on rank 0 (callers weight / log dropouts; non-roots
    /// get the averaged `data` and a count of 0). This is how a slow or
    /// offline edge client can't stall a training round.
    pub fn federated_average(
        &self,
        data: &mut [f32],
        deadline: Duration,
    ) -> Result<usize, CollectiveError> {
        let n = self.world_size();
        if n <= 1 {
            return Ok(1);
        }
        let elems = data.len();
        if self.rank() == 0 {
            let mut acc = data.to_vec();
            let mut present = 1usize;
            let end = Instant::now() + deadline;
            for r in 1..n {
                let remaining = end.saturating_duration_since(Instant::now());
                if let Some(bytes) =
                    self.transport
                        .recv_bytes_timeout(r, TAG_ALL_REDUCE, remaining)?
                {
                    let other = le_bytes_to_f32(&bytes)?;
                    if other.len() == elems {
                        for i in 0..elems {
                            acc[i] += other[i];
                        }
                        present += 1;
                    }
                }
            }
            let inv = 1.0 / present as f32;
            for v in acc.iter_mut() {
                *v *= inv;
            }
            data.copy_from_slice(&acc);
            for r in 1..n {
                self.send_f32_tagged(r, TAG_ALL_REDUCE, &acc)?;
            }
            Ok(present)
        } else {
            self.send_f32_tagged(0, TAG_ALL_REDUCE, data)?;
            let res = self.recv_f32_tagged(0, TAG_ALL_REDUCE)?;
            if res.len() == elems {
                data.copy_from_slice(&res);
            }
            Ok(0)
        }
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

    fn f16_bytes(vals: &[f32]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect()
    }
    fn f16_vals(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()
    }

    #[test]
    fn all_reduce_typed_f16_sums_across_ranks() {
        run_ranks(4, |g| {
            let mut bytes = f16_bytes(&vec![g.rank() as f32 + 1.0; 3]); // 1,2,3,4
            g.all_reduce_typed(&mut bytes, DType::F16, ReduceKind::Sum)
                .unwrap();
            assert_eq!(f16_vals(&bytes), vec![10.0; 3], "rank {}", g.rank());
        });
    }

    #[test]
    fn all_reduce_typed_bf16_mean() {
        run_ranks(4, |g| {
            let mut bytes: Vec<u8> = vec![g.rank() as f32 + 1.0; 4]
                .iter()
                .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
                .collect();
            g.all_reduce_typed(&mut bytes, DType::BF16, ReduceKind::Mean)
                .unwrap();
            let got: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect();
            // mean(1,2,3,4) = 2.5, exactly representable in bf16
            assert_eq!(got, vec![2.5; 4], "rank {}", g.rank());
        });
    }

    #[test]
    fn all_reduce_typed_uneven_length() {
        // 5 elems over 4 ranks exercises the remainder chunking.
        run_ranks(4, |g| {
            let mut bytes = f16_bytes(&vec![g.rank() as f32 + 1.0; 5]);
            g.all_reduce_typed(&mut bytes, DType::F16, ReduceKind::Sum)
                .unwrap();
            assert_eq!(f16_vals(&bytes), vec![10.0; 5], "rank {}", g.rank());
        });
    }

    #[test]
    fn all_reduce_typed_f16_large_buffer_simd_path() {
        // 1000 elems is past the SIMD-batch threshold, so this exercises the
        // vectorized `convert_*_f32_slice` path (small tests use it too, but
        // this guards the batched conversion against off-by-one regressions).
        run_ranks(3, |g| {
            let mut bytes = f16_bytes(&vec![g.rank() as f32 + 1.0; 1000]); // 1,2,3
            g.all_reduce_typed(&mut bytes, DType::F16, ReduceKind::Sum)
                .unwrap();
            assert_eq!(f16_vals(&bytes), vec![6.0; 1000], "rank {}", g.rank());
        });
    }

    #[test]
    fn all_reduce_typed_i8_federated_mean() {
        // The edge case: three int8 "clients" averaged (FedAvg-style).
        run_ranks(3, |g| {
            let mut bytes = vec![((g.rank() + 1) * 10) as i8 as u8; 4]; // 10,20,30
            g.all_reduce_typed(&mut bytes, DType::I8, ReduceKind::Mean)
                .unwrap();
            let got: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
            assert_eq!(got, vec![20i8; 4], "rank {}", g.rank()); // mean = 20
        });
    }

    #[test]
    fn all_reduce_typed_i8_sum_saturates() {
        run_ranks(2, |g| {
            let mut bytes = vec![100i8 as u8; 2];
            g.all_reduce_typed(&mut bytes, DType::I8, ReduceKind::Sum)
                .unwrap();
            let got: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
            assert_eq!(got, vec![127i8; 2], "rank {}", g.rank()); // 200 clamps to i8::MAX
        });
    }

    #[test]
    fn negotiate_reduce_dtype_rules() {
        assert_eq!(
            negotiate_reduce_dtype(&[DType::F16, DType::BF16]),
            DType::F32
        );
        assert_eq!(
            negotiate_reduce_dtype(&[DType::F16, DType::F64]),
            DType::F64
        );
        assert_eq!(negotiate_reduce_dtype(&[]), DType::F32);
    }
}
