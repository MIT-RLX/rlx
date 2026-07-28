// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Qwen2-VL window-attention token reorder helpers.
//!
//! Matches llama.cpp `ggml_get_rows` on a 2-D matrix after packing `merge_sq`
//! spatial tokens into the feature axis. Sequence length stays `n_pos`; only
//! the `n_pos / merge_sq` window groups are permuted.

use super::{HirGraphExt, HirMut, HirNodeId};

/// Permute window groups before window-attention blocks.
///
/// Input `x`: `[batch, n_pos, n_embd]`  
/// `inv_window_idx`: `[n_win]` with `n_win = n_pos / merge_sq`  
/// Output: `[batch, n_pos, n_embd]`
pub fn window_token_gather_bsn(
    g: &mut HirMut,
    x: HirNodeId,
    inv_window_idx: HirNodeId,
    batch: usize,
    n_pos: usize,
    n_embd: usize,
    merge_sq: usize,
) -> HirNodeId {
    debug_assert!(merge_sq > 0 && n_pos.is_multiple_of(merge_sq));
    let n_win = n_pos / merge_sq;
    let feat = n_embd * merge_sq;
    let packed = g.reshape_(x, vec![batch as i64, n_win as i64, feat as i64]);
    let flat = g.reshape_(packed, vec![n_win as i64, feat as i64]);
    let gathered = g.gather_(flat, inv_window_idx, 0);
    let grouped = g.reshape_(gathered, vec![batch as i64, n_win as i64, feat as i64]);
    g.reshape_(grouped, vec![batch as i64, n_pos as i64, n_embd as i64])
}

/// Restore window order after the spatial merger MLP.
///
/// Input `x`: `[batch, n_out, n_embd]` (or equivalent 2-D `[n_out, n_embd]`)  
/// `window_idx`: `[n_out]`  
/// Output: `[batch, n_out, n_embd]`
pub fn window_token_scatter_bsn(
    g: &mut HirMut,
    x: HirNodeId,
    window_idx: HirNodeId,
    batch: usize,
    n_out: usize,
    n_embd: usize,
) -> HirNodeId {
    let flat = g.reshape_(x, vec![n_out as i64, n_embd as i64]);
    let gathered = g.gather_(flat, window_idx, 0);
    g.reshape_(gathered, vec![batch as i64, n_out as i64, n_embd as i64])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, HirModule, Shape};

    #[test]
    fn window_gather_preserves_n_pos() {
        let batch = 1usize;
        let n_pos = 1600usize;
        let n_embd = 1280usize;
        let merge_sq = 4usize;
        let n_win = n_pos / merge_sq;
        let f = DType::F32;

        let mut hir = HirModule::new("window_gather");
        let x = hir.input("x", Shape::new(&[batch, n_pos, n_embd], f));
        let inv = hir.input("inv", Shape::new(&[n_win], f));
        let mut g = HirMut::new(&mut hir);
        let out = window_token_gather_bsn(&mut g, x, inv, batch, n_pos, n_embd, merge_sq);
        let shape = hir.node(out).shape.clone();
        assert_eq!(shape, Shape::new(&[batch, n_pos, n_embd], f));
    }

    #[test]
    fn window_scatter_keeps_n_out() {
        let batch = 1usize;
        let n_out = 400usize;
        let n_embd = 3584usize;
        let f = DType::F32;

        let mut hir = HirModule::new("window_scatter");
        let x = hir.input("x", Shape::new(&[batch, n_out, n_embd], f));
        let win = hir.input("win", Shape::new(&[n_out], f));
        let mut g = HirMut::new(&mut hir);
        let out = window_token_scatter_bsn(&mut g, x, win, batch, n_out, n_embd);
        let shape = hir.node(out).shape.clone();
        assert_eq!(shape, Shape::new(&[batch, n_out, n_embd], f));
    }
}
