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

//! Async tile-copy + double-buffer primitives (plan #22).
//!
//! Borrowed from MAX's
//! `layout/{tma_async, tensor_core_async}.mojo` +
//! `structured_kernels/{pipeline, pipeline_storage, barriers}.mojo`.
//! On NVIDIA the equivalent is TMA (Tensor Memory Accelerator);
//! on Apple Silicon there's no direct analog because the GPU and
//! CPU share a unified memory pool — but the *pipelining* idea
//! still pays off: while shader N runs on tile N, you issue an
//! async copy / blit for tile N+1 and let the two overlap.
//!
//! The shape this module exposes:
//!
//!   - [`DoubleBuffer<T>`] — owns two `T` instances with a `swap`
//!     pointer; `current()` is what compute reads, `next_mut()` is
//!     where the async copy lands.
//!   - [`AsyncCopy`] trait — `issue()` schedules a copy and returns
//!     a [`BarrierToken`]; `wait()` blocks until the matching
//!     issue has completed.
//!   - [`SyncCopy`] — the CPU implementation: every issue is a
//!     memcpy + a fresh token; `wait()` is a no-op (the copy
//!     already completed). Sufficient for unit tests and for
//!     bench harnesses that run the pipeline pattern with no
//!     real overlap.
//!
//! A future Metal impl plugs in via the same trait. The Metal
//! version would issue a `MTLBlitCommandEncoder.copy(...)` on a
//! distinct command queue and signal an `MTLEvent` for `wait()`.

use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque ticket returned by [`AsyncCopy::issue`]. Pass back to
/// [`AsyncCopy::wait`] to block until the corresponding copy is
/// done. Tokens are scoped to one engine — don't pass them across.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarrierToken(pub u64);

/// Pluggable async-copy engine. Backends (`SyncCopy` for CPU,
/// future `MetalBlitCopy` for GPU) implement this.
pub trait AsyncCopy {
    /// Schedule a `bytes`-byte copy from `src` to `dst`. Returns a
    /// token usable with [`Self::wait`].
    /// # Safety
    /// `src` valid for read, `dst` valid for write, `bytes` doesn't
    /// overflow either region. Caller ensures `src` and `dst` don't
    /// alias unless that's intentional.
    unsafe fn issue(&mut self, src: *const u8, dst: *mut u8, bytes: usize) -> BarrierToken;

    /// Block until the copy referred to by `token` has completed.
    fn wait(&mut self, token: BarrierToken);
}

/// CPU "async" copy — actually synchronous. `issue()` does a
/// `memcpy` immediately and returns a fresh token; `wait()` is a
/// no-op. Useful as the test fixture and for code paths that
/// don't actually need overlap.
pub struct SyncCopy {
    counter: AtomicU64,
}

impl SyncCopy {
    pub const fn new() -> Self {
        Self {
            counter: AtomicU64::new(0),
        }
    }
}

impl Default for SyncCopy {
    fn default() -> Self {
        Self::new()
    }
}

impl AsyncCopy for SyncCopy {
    unsafe fn issue(&mut self, src: *const u8, dst: *mut u8, bytes: usize) -> BarrierToken {
        unsafe {
            std::ptr::copy_nonoverlapping(src, dst, bytes);
        }
        BarrierToken(self.counter.fetch_add(1, Ordering::Relaxed))
    }

    fn wait(&mut self, _token: BarrierToken) {
        // Sync copy: already done at issue() time.
    }
}

/// Two-buffer ring. `current()` is what compute reads this step;
/// `next_mut()` is where the *next* async copy should land. Call
/// `swap()` after waiting on the current copy to advance.
#[derive(Debug, Clone)]
pub struct DoubleBuffer<T> {
    buffers: [T; 2],
    active: usize,
}

impl<T> DoubleBuffer<T> {
    pub fn new(a: T, b: T) -> Self {
        Self {
            buffers: [a, b],
            active: 0,
        }
    }

    pub fn current(&self) -> &T {
        &self.buffers[self.active]
    }
    pub fn current_mut(&mut self) -> &mut T {
        &mut self.buffers[self.active]
    }

    pub fn next(&self) -> &T {
        &self.buffers[1 - self.active]
    }
    pub fn next_mut(&mut self) -> &mut T {
        &mut self.buffers[1 - self.active]
    }

    /// Flip which buffer is current. Typical pattern:
    /// ```text
    /// // At step k:
    /// engine.wait(prev_token);          // copy of tile-k done
    /// let token_for_kp1 = engine.issue(src_kp1, double.next_mut(), bytes);
    /// compute(double.current());        // shader runs on tile-k
    /// double.swap();                    // tile-(k+1) becomes current
    /// // → at step k+1, wait(token_for_kp1) etc.
    /// ```
    pub fn swap(&mut self) {
        self.active = 1 - self.active;
    }

    /// Both buffers' shared length, when `T = Vec<u8>` / `Vec<f32>`.
    /// Exposed for symmetry; many callers don't need it.
    pub fn pair(&self) -> (&T, &T) {
        (&self.buffers[0], &self.buffers[1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_buffer_swap_round_trip() {
        let mut db = DoubleBuffer::new(vec![1u8; 4], vec![2u8; 4]);
        assert_eq!(db.current(), &vec![1u8; 4]);
        db.swap();
        assert_eq!(db.current(), &vec![2u8; 4]);
        db.swap();
        assert_eq!(db.current(), &vec![1u8; 4]);
    }

    #[test]
    fn sync_copy_round_trips_data() {
        let src = [1u8, 2, 3, 4];
        let mut dst = [0u8; 4];
        let mut engine = SyncCopy::new();
        let token = unsafe { engine.issue(src.as_ptr(), dst.as_mut_ptr(), 4) };
        engine.wait(token);
        assert_eq!(dst, src);
    }

    #[test]
    fn pipelined_pattern_through_double_buffer() {
        // Simulate the canonical compute-overlap-copy loop:
        //   tile 0..N comes in two halves [0..2] and [2..N]; the
        //   compute step is "sum of the buffer".
        let source: Vec<u8> = (0..16u8).collect();
        let tile_bytes = 4;
        let mut db = DoubleBuffer::new(vec![0u8; tile_bytes], vec![0u8; tile_bytes]);
        let mut engine = SyncCopy::new();

        // Prime: load tile 0 into the *current* slot.
        let t0 =
            unsafe { engine.issue(source.as_ptr(), db.current_mut().as_mut_ptr(), tile_bytes) };
        engine.wait(t0);

        let mut total: u64 = 0;
        let mut tile_idx = 1usize;
        while tile_idx * tile_bytes < source.len() {
            // Issue copy for next tile into the inactive slot.
            let t = unsafe {
                engine.issue(
                    source.as_ptr().add(tile_idx * tile_bytes),
                    db.next_mut().as_mut_ptr(),
                    tile_bytes,
                )
            };
            // Compute on the current tile.
            total += db.current().iter().map(|&b| b as u64).sum::<u64>();
            // Step boundary.
            engine.wait(t);
            db.swap();
            tile_idx += 1;
        }
        // Drain the last tile.
        total += db.current().iter().map(|&b| b as u64).sum::<u64>();

        // Sum of 0..16 = 120.
        let expected: u64 = (0..16u64).sum();
        assert_eq!(total, expected);
    }
}
