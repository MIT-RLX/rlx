// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Slice specs — view-like narrowing without materializing data.

use rlx_ir::Dim;

use crate::Tensor;

/// One axis in a slice (`s![ax(), rg(2, 10)]` or `s![.., 2..10]` via helpers).
///
/// Indices are signed and NumPy-style: negative values count from the end
/// (`-1` = last), resolved against the axis's (static) size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SliceAxis {
    All,
    Index(i64),
    Range { start: i64, end: i64 },
    From(i64),
}

/// Keep the full axis.
#[inline]
pub fn ax() -> SliceAxis {
    SliceAxis::All
}

/// Single index (negative counts from the end). Keeps the axis as size 1 —
/// use [`Tensor::select`](crate::Tensor::select) to drop it.
#[inline]
pub fn ix(i: i64) -> SliceAxis {
    SliceAxis::Index(i)
}

/// Half-open range `start..end` (either bound may be negative).
#[inline]
pub fn rg(start: i64, end: i64) -> SliceAxis {
    SliceAxis::Range { start, end }
}

/// `start..` through the end of a static axis (negative `start` allowed).
#[inline]
pub fn tail(start: i64) -> SliceAxis {
    SliceAxis::From(start)
}

/// Resolve a signed index against an axis size (negative → `size + idx`).
/// Panics on a dynamic axis or out-of-range index.
pub(crate) fn resolve_index(idx: i64, dim: Option<usize>, axis: usize) -> usize {
    if idx >= 0 {
        return idx as usize;
    }
    let n = dim
        .unwrap_or_else(|| panic!("negative index requires a static dimension on axis {axis}"))
        as i64;
    let resolved = n + idx;
    assert!(
        resolved >= 0,
        "index {idx} out of bounds for axis {axis} of size {n}"
    );
    resolved as usize
}

/// Multi-axis slice. Apply via [`Tensor::slice`].
#[derive(Debug, Clone, Default)]
pub struct SliceSpec {
    axes: Vec<SliceAxis>,
}

impl SliceSpec {
    pub fn new(axes: Vec<SliceAxis>) -> Self {
        Self { axes }
    }

    /// Lower to a chain of `narrow` ops (compiler may fuse into consumers).
    pub fn apply(&self, tensor: &Tensor) -> Tensor {
        let mut out = tensor.clone();
        let shape = tensor.shape();
        for (axis, spec) in self.axes.iter().enumerate() {
            let dim = match shape.dims().get(axis) {
                Some(Dim::Static(n)) => Some(*n),
                Some(Dim::Dynamic(_)) => None,
                None => panic!("slice axis {axis} out of bounds (rank {})", shape.rank()),
            };
            out = match spec {
                SliceAxis::All => out,
                SliceAxis::Index(i) => out.narrow(axis, resolve_index(*i, dim, axis), 1),
                SliceAxis::Range { start, end } => {
                    let s = resolve_index(*start, dim, axis);
                    let e = resolve_index(*end, dim, axis);
                    assert!(e > s, "slice range end must exceed start (axis {axis})");
                    out.narrow(axis, s, e - s)
                }
                SliceAxis::From(start) => {
                    let s = resolve_index(*start, dim, axis);
                    let n = dim.unwrap_or_else(|| {
                        panic!("slice `start..` requires a static dimension on axis {axis}")
                    });
                    out.narrow(axis, s, n - s)
                }
            };
        }
        out
    }
}

/// Build a [`SliceSpec`] from axis expressions.
///
/// ```rust
/// use rlx_tensor::{ax, graph, rg, s, shape};
///
/// let g = graph("win", |g| {
///     let x = g.input("x", shape![4, 16]);
///     x.slice(s![ax(), rg(2, 10)])
/// });
/// assert_eq!(g.outputs.len(), 1);
/// ```
#[macro_export]
macro_rules! s {
    ( $( $axis:expr ),* $(,)? ) => {
        $crate::SliceSpec::new(vec![ $( $axis ),* ])
    };
}
