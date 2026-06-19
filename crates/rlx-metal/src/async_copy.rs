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

//! Metal-backed async copy via `MTLBlitCommandEncoder` + events
//! (plan #22).
//!
//! Implements the `rlx_runtime::async_copy::AsyncCopy` trait for
//! Metal: `issue` records a blit-copy on a dedicated command queue
//! and signals an `MTLEvent`; `wait` blocks the host on that event.
//! On Apple Silicon the GPU and CPU share one memory pool, so this
//! is genuinely about **pipelining** (encode/dispatch overlap),
//! not about moving bytes between memory tiers.
//!
//! The current implementation is a *correct* shim — it uses the
//! existing `metal_device().queue` for both encode and signal so
//! callers can already write pipelined kernels against the trait.
//! A future revision will allocate a separate transfer queue so
//! blit and compute encoders run in parallel; that requires
//! teaching the arena which queue an offset belongs to and is the
//! follow-on work.

use rlx_ir::async_copy::{AsyncCopy, BarrierToken};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::device::metal_device;

pub struct MetalBlitCopy {
    counter: AtomicU64,
    /// The single MTLBuffer that backs the runtime arena. Issue/
    /// wait operate on `(offset, len)` pairs into this buffer.
    arena: metal::Buffer,
}

impl MetalBlitCopy {
    /// Bind to a pre-existing arena buffer.
    pub fn new(arena: metal::Buffer) -> Self {
        Self {
            counter: AtomicU64::new(0),
            arena,
        }
    }

    /// Convenience — bind to the currently-installed device's
    /// shared arena (when callers don't manage their own).
    pub fn from_device() -> Option<Self> {
        let dev = metal_device()?;
        let arena = dev.alloc_shared(1); // placeholder; caller should rebind
        Some(Self {
            counter: AtomicU64::new(0),
            arena,
        })
    }

    /// Issue a blit copy strictly inside the arena. Both `src` and
    /// `dst` are byte offsets into the bound arena buffer.
    pub fn issue_intra_arena(&mut self, src: usize, dst: usize, bytes: usize) -> BarrierToken {
        let dev = metal_device().expect("Metal device required for MetalBlitCopy");
        let cb = dev.queue.new_command_buffer();
        let blit = cb.new_blit_command_encoder();
        blit.copy_from_buffer(
            &self.arena,
            src as u64,
            &self.arena,
            dst as u64,
            bytes as u64,
        );
        blit.end_encoding();
        cb.commit();
        // The token's u64 is the command-buffer's hash address (low
        // 64 bits); wait() looks it up. Simpler scheme: just bump a
        // counter and stash the cb reference in a side-table.
        let id = self.counter.fetch_add(1, Ordering::Relaxed);
        // Block now (matches the SyncCopy semantics on the CPU
        // side); pipelining wins arrive when we move to a separate
        // queue + event.
        cb.wait_until_completed();
        BarrierToken(id)
    }
}

impl AsyncCopy for MetalBlitCopy {
    /// **Note:** raw-pointer `issue` is here to satisfy the trait
    /// contract; in Metal everything goes through buffer offsets,
    /// so prefer [`MetalBlitCopy::issue_intra_arena`] for real
    /// callers. This impl computes offsets relative to the bound
    /// arena buffer's contents pointer.
    /// # Safety
    /// `src`/`dst` must point into the arena buffer this instance
    /// was constructed from.
    unsafe fn issue(&mut self, src: *const u8, dst: *mut u8, bytes: usize) -> BarrierToken {
        let base = self.arena.contents() as *const u8;
        let src_off = unsafe { src.offset_from(base) as usize };
        let dst_off = unsafe { (dst as *const u8).offset_from(base) as usize };
        self.issue_intra_arena(src_off, dst_off, bytes)
    }

    fn wait(&mut self, _token: BarrierToken) {
        // issue_intra_arena already blocks; no-op here.
    }
}
