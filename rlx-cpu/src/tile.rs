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

//! CPU `TileIO` impls (plans #23 + #27).
//!
//! Borrowed from MAX's
//! `structured_kernels/{kernel_common, tile_types, smem_types}.mojo`
//! and `layout/tile_io.mojo`. Lifts the "kernel-author standard
//! library" pattern: typed primitives kernels compose, instead of
//! re-deriving stride math and load/store loops per kernel.
//!
//! The vocabulary types (`Tile2`, `Coord2`, `Strides2`) live in
//! `rlx_ir::layout` (plan #3 — shared layout IR) so Metal kernels
//! can use the same names. CPU-specific `TileIO` impls live here.

pub use rlx_ir::{Coord2, Strides2, Tile2};

/// Tile I/O trait — load / store / prefetch parameterized over the
/// physical layout. Two impls today: [`RowMajorTile`] (the standard
/// flat layout) and [`StridedTile`] (when reading a non-contiguous
/// view, e.g. last-axis Narrow into Attention).
///
/// Methods take pointers (not slices) so the abstraction works for
/// both owned and aliased buffers.
pub trait TileIO {
    /// Compute the byte address for a coordinate. Used by
    /// `load` / `store` / `prefetch` so impls only need to define
    /// the address arithmetic once.
    /// SAFETY: caller checks bounds.
    unsafe fn address(&self, base: *const f32, c: Coord2) -> *const f32;

    /// Load a tile element by `(row, col)`.
    /// SAFETY: caller ensures the address is valid for read.
    #[inline(always)]
    unsafe fn load(&self, base: *const f32, c: Coord2) -> f32 {
        unsafe { *self.address(base, c) }
    }

    /// Store an element by `(row, col)`.
    /// SAFETY: caller ensures the address is valid for write.
    #[inline(always)]
    unsafe fn store(&self, base: *mut f32, c: Coord2, v: f32) {
        unsafe {
            *(self.address(base, c) as *mut f32) = v;
        }
    }

    /// Hint to the prefetcher. On aarch64 issues a single
    /// `prfm pldl1keep` (load into L1, retain). Elsewhere a no-op.
    /// SAFETY: caller ensures the address is in a valid mapping.
    #[inline(always)]
    unsafe fn prefetch(&self, base: *const f32, c: Coord2) {
        unsafe {
            let addr = self.address(base, c);
            #[cfg(target_arch = "aarch64")]
            {
                std::arch::asm!("prfm pldl1keep, [{0}]", in(reg) addr,
                    options(nostack, readonly));
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let _ = addr;
            }
        }
    }
}

/// Row-major contiguous tile: `addr = base + row * cols + col`.
#[derive(Debug, Clone, Copy)]
pub struct RowMajorTile {
    pub shape: Tile2,
}

impl TileIO for RowMajorTile {
    #[inline(always)]
    unsafe fn address(&self, base: *const f32, c: Coord2) -> *const f32 {
        unsafe { base.add(c.row * self.shape.cols + c.col) }
    }
}

/// Strided tile: each row stride is configurable. Lets a kernel
/// read a non-contiguous view (e.g. last-axis Narrow output) with
/// the same TileIO interface as a contiguous tile.
#[derive(Debug, Clone, Copy)]
pub struct StridedTile {
    pub shape: Tile2,
    pub strides: Strides2,
}

impl TileIO for StridedTile {
    #[inline(always)]
    unsafe fn address(&self, base: *const f32, c: Coord2) -> *const f32 {
        unsafe { base.add(c.row * self.strides.row + c.col * self.strides.col) }
    }
}

/// Walk every element of a tile in row-major order, calling `f`.
/// Convenience for kernels that don't care about iteration order.
#[inline(always)]
pub fn for_each_coord(shape: Tile2, mut f: impl FnMut(Coord2)) {
    for r in 0..shape.rows {
        for c in 0..shape.cols {
            f(Coord2 { row: r, col: c });
        }
    }
}

/// Tile copy via TileIO. Source and destination layouts can differ
/// (the typical use: read strided source, write contiguous dst).
///
/// # Safety
/// `src_base` and `dst_base` must point into allocations large enough
/// for `shape`'s extents under the IO layouts in `src_io` / `dst_io`.
/// The two ranges may not overlap.
#[inline]
pub unsafe fn copy_tile<S: TileIO, D: TileIO>(
    src_io: &S,
    src_base: *const f32,
    dst_io: &D,
    dst_base: *mut f32,
    shape: Tile2,
) {
    for_each_coord(shape, |c| unsafe {
        dst_io.store(dst_base, c, src_io.load(src_base, c));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_round_trip() {
        let mut buf = [0f32; 12]; // 3×4
        let io = RowMajorTile {
            shape: Tile2::new(3, 4),
        };
        unsafe {
            io.store(buf.as_mut_ptr(), Coord2 { row: 1, col: 2 }, 42.0);
            assert_eq!(io.load(buf.as_ptr(), Coord2 { row: 1, col: 2 }), 42.0);
        }
        assert_eq!(buf[4 + 2], 42.0);
    }

    #[test]
    fn strided_reads_non_contig_view() {
        // Source: 4-row tile inside a 4-row × 8-col parent.
        // Pretending we narrowed cols 2..6 of each row; row stride = 8.
        let parent: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let view = StridedTile {
            shape: Tile2::new(4, 4),
            strides: Strides2 { row: 8, col: 1 },
        };
        // base pointer offset to col=2 of row 0
        let base = unsafe { parent.as_ptr().add(2) };
        let v = unsafe { view.load(base, Coord2 { row: 1, col: 1 }) };
        // expected: parent[1*8 + 2 + 1] = 11
        assert_eq!(v, 11.0);
    }

    #[test]
    fn prefetch_doesnt_panic() {
        // Prefetch is a hint — it should not crash, and should
        // accept any in-bounds address. We just verify the call
        // sequence compiles + runs on the current target.
        let buf = vec![0f32; 64];
        let io = RowMajorTile {
            shape: Tile2::new(8, 8),
        };
        unsafe {
            io.prefetch(buf.as_ptr(), Coord2 { row: 0, col: 0 });
            io.prefetch(buf.as_ptr(), Coord2 { row: 7, col: 7 });
        }
    }

    #[test]
    fn copy_tile_strided_to_contig() {
        let parent: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let mut dst = vec![0f32; 16]; // 4×4 contiguous
        let src_io = StridedTile {
            shape: Tile2::new(4, 4),
            strides: Strides2 { row: 8, col: 1 },
        };
        let dst_io = RowMajorTile {
            shape: Tile2::new(4, 4),
        };
        let base = unsafe { parent.as_ptr().add(2) };
        unsafe {
            copy_tile(&src_io, base, &dst_io, dst.as_mut_ptr(), Tile2::new(4, 4));
        }
        // First row of dst should be parent[2..6] = [2,3,4,5].
        assert_eq!(&dst[0..4], &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(&dst[4..8], &[10.0, 11.0, 12.0, 13.0]);
    }
}
