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

//! Shared layout vocabulary (plan #3).
//!
//! Tile / coordinate / stride types used by every kernel-author
//! crate. Lives in `rlx-ir` (the leaf) so CPU and Metal stop
//! re-deriving stride math independently.
//!
//! Backend-specific I/O (CPU pointer reads, Metal threadgroup
//! loads) lives in the backend's own crate behind a `TileIO` trait
//! — only the *vocabulary* is shared here.

/// 2-D row-major or strided tile shape (in elements).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile2 {
    pub rows: usize,
    pub cols: usize,
}

impl Tile2 {
    pub const fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols }
    }
    pub const fn area(self) -> usize {
        self.rows * self.cols
    }
}

/// 2-D coordinate within a tile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Coord2 {
    pub row: usize,
    pub col: usize,
}

/// Per-axis strides in **elements** (not bytes). `row` is the
/// distance between consecutive rows; `col` between consecutive
/// columns. For a contiguous row-major tile of shape (R, C):
/// `Strides2 { row: C, col: 1 }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strides2 {
    pub row: usize,
    pub col: usize,
}

impl Strides2 {
    pub const fn row_major(cols: usize) -> Self {
        Self { row: cols, col: 1 }
    }
    pub const fn col_major(rows: usize) -> Self {
        Self { row: 1, col: rows }
    }
}

/// Hierarchical shape tuple (plan #38). Borrowed from MAX's
/// `layout/int_tuple.mojo`: shapes nest, so a `((B, S), (H, D))`
/// expression captures the "outer batch+seq, inner heads+head_dim"
/// structure of a tiled layout. Useful for kernels that want to
/// reason about block-tiled sweeps without re-deriving the
/// implied stride math each time.
///
/// Stays alongside the existing flat [`crate::Shape`] (which is
/// what every op carries today). New code that benefits from
/// hierarchy uses [`ShapeTuple`]; we don't migrate Shape because
/// the entire codebase is built around it and the win is
/// concentrated in advanced layout / fusion code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeTuple {
    /// Single concrete dimension.
    Leaf(usize),
    /// Ordered list of sub-tuples. Nesting is unbounded.
    Nested(Vec<ShapeTuple>),
}

impl ShapeTuple {
    /// One-dim leaf. `ShapeTuple::leaf(8)`.
    pub fn leaf(n: usize) -> Self {
        Self::Leaf(n)
    }

    /// Wrapping constructor for nested layouts.
    pub fn nested(parts: Vec<ShapeTuple>) -> Self {
        Self::Nested(parts)
    }

    /// Convenience: build a flat tuple from `&[usize]`. Each
    /// element becomes a `Leaf`. `flat(&[2, 3, 4])` is equivalent
    /// to `Nested(vec![Leaf(2), Leaf(3), Leaf(4)])`.
    pub fn flat(dims: &[usize]) -> Self {
        Self::Nested(dims.iter().map(|&n| Self::Leaf(n)).collect())
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, Self::Leaf(_))
    }

    /// Top-level rank. Leaves are rank 1; nested tuples are the
    /// length of the outer list.
    pub fn rank(&self) -> usize {
        match self {
            Self::Leaf(_) => 1,
            Self::Nested(v) => v.len(),
        }
    }

    /// Total element count, traversing the entire hierarchy.
    pub fn product(&self) -> usize {
        match self {
            Self::Leaf(n) => *n,
            Self::Nested(v) => v.iter().map(|p| p.product()).product(),
        }
    }

    /// Flatten into a row-major sequence of leaves. Useful when
    /// converting to the existing `Shape` type.
    pub fn flatten(&self) -> Vec<usize> {
        let mut out = Vec::new();
        self.flatten_into(&mut out);
        out
    }

    fn flatten_into(&self, out: &mut Vec<usize>) {
        match self {
            Self::Leaf(n) => out.push(*n),
            Self::Nested(v) => v.iter().for_each(|p| p.flatten_into(out)),
        }
    }

    /// Walk a path of indices through the hierarchy. Returns
    /// the sub-tuple at `path` or `None` if the path goes out of
    /// bounds at any level.
    ///
    /// `[]` returns `Some(self)`; `[0]` returns the first child.
    pub fn get(&self, path: &[usize]) -> Option<&ShapeTuple> {
        if path.is_empty() {
            return Some(self);
        }
        match self {
            Self::Leaf(_) => None, // can't descend into a leaf
            Self::Nested(v) => v.get(path[0]).and_then(|c| c.get(&path[1..])),
        }
    }
}

/// Ragged-tensor descriptor (plan #4). Represents a tensor of
/// variable-length sequences laid out without padding:
///
///   data:    [total_elems, trailing_dim]   flat
///   offsets: [batch + 1]                    cumulative starts
///
/// `data[offsets[i]..offsets[i+1]]` is row `i`'s contents (each
/// row has `(offsets[i+1] - offsets[i])` elements times trailing).
///
/// Borrowed from MAX's `nn/_ragged_utils.mojo`, `kv_cache_ragged.mojo`,
/// and `gemv_partial_norm.mojo`. Essential for serving throughput when
/// sequences in a batch have very different lengths — padding to max
/// wastes most of the work; ragged + offset-driven kernels process each
/// row at its actual length.
///
/// Today this is the type vocabulary; kernel paths come per-op as
/// the ragged use-case lands (the cumsum primitive #44 already
/// covers offset construction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ragged {
    /// Number of rows (= batch).
    pub rows: usize,
    /// Trailing per-element width. For BERT it's the hidden
    /// dimension; for KV cache it's `num_heads * head_dim`. 1 if
    /// the tensor is a flat sequence of scalars.
    pub trailing: usize,
    /// Total elements across all rows (sum of per-row lengths).
    /// Equals `offsets[rows]` when offsets are materialized.
    pub total: usize,
}

impl Ragged {
    pub const fn new(rows: usize, trailing: usize, total: usize) -> Self {
        Self {
            rows,
            trailing,
            total,
        }
    }

    /// Total f32 element count (data) — does not count the offsets
    /// table.
    pub const fn data_elements(self) -> usize {
        self.total * self.trailing
    }

    /// Element count of the offsets table (`rows + 1`).
    pub const fn offsets_elements(self) -> usize {
        self.rows + 1
    }
}

/// 3-D extension for `[batch, rows, cols]` tiles. Common for
/// per-head attention sweeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile3 {
    pub batch: usize,
    pub rows: usize,
    pub cols: usize,
}

impl Tile3 {
    pub const fn new(batch: usize, rows: usize, cols: usize) -> Self {
        Self { batch, rows, cols }
    }
    pub const fn area(self) -> usize {
        self.batch * self.rows * self.cols
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Strides3 {
    pub batch: usize,
    pub row: usize,
    pub col: usize,
}

impl Strides3 {
    pub const fn row_major(rows: usize, cols: usize) -> Self {
        Self {
            batch: rows * cols,
            row: cols,
            col: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile2_area() {
        assert_eq!(Tile2::new(3, 4).area(), 12);
    }

    #[test]
    fn strides2_presets() {
        assert_eq!(Strides2::row_major(8), Strides2 { row: 8, col: 1 });
        assert_eq!(Strides2::col_major(8), Strides2 { row: 1, col: 8 });
    }

    #[test]
    fn strides3_row_major() {
        assert_eq!(
            Strides3::row_major(3, 4),
            Strides3 {
                batch: 12,
                row: 4,
                col: 1
            }
        );
    }

    // Tuple tests live here so `tuple` test names cover the new
    // hierarchical type (the runtime smoke covers the const fns).
    #[test]
    fn tuple_leaf_constructors() {
        let a = ShapeTuple::leaf(8);
        assert_eq!(a.flatten(), vec![8]);
        assert_eq!(a.product(), 8);
        assert!(a.is_leaf());
    }

    #[test]
    fn tuple_flat_constructor() {
        let s = ShapeTuple::flat(&[2, 3, 4]);
        assert_eq!(s.flatten(), vec![2, 3, 4]);
        assert_eq!(s.product(), 24);
        assert_eq!(s.rank(), 3);
    }

    #[test]
    fn tuple_nested_product_and_flatten() {
        // BERT-shape: ((batch, seq), (heads, head_dim)).
        let bs = ShapeTuple::nested(vec![ShapeTuple::leaf(8), ShapeTuple::leaf(15)]);
        let nh = ShapeTuple::nested(vec![ShapeTuple::leaf(12), ShapeTuple::leaf(64)]);
        let s = ShapeTuple::nested(vec![bs, nh]);
        assert_eq!(s.product(), 8 * 15 * 12 * 64);
        assert_eq!(s.flatten(), vec![8, 15, 12, 64]);
        assert_eq!(s.rank(), 2); // top-level rank
    }

    #[test]
    fn tuple_get_resolves_path() {
        let inner = ShapeTuple::nested(vec![ShapeTuple::leaf(12), ShapeTuple::leaf(64)]);
        let s = ShapeTuple::nested(vec![ShapeTuple::leaf(8), ShapeTuple::leaf(15), inner]);
        assert_eq!(s.get(&[0]), Some(&ShapeTuple::Leaf(8)));
        assert_eq!(s.get(&[2, 1]), Some(&ShapeTuple::Leaf(64)));
        assert_eq!(s.get(&[2, 99]), None);
    }

    #[test]
    fn ragged_element_counts() {
        // 4 rows with total 30 elements; trailing = 8 (hidden dim).
        let r = Ragged::new(4, 8, 30);
        assert_eq!(r.data_elements(), 240); // 30 * 8 floats
        assert_eq!(r.offsets_elements(), 5); // rows + 1
    }
}
