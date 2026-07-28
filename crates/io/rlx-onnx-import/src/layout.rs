// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Rank-3 layout helpers for ONNX graphs (BLC vs seq-first `[seq, 1, C]`).

use rlx_ir::Shape;

/// Channel width typical of transformer / LSTM blocks.
pub fn is_typical_channel(n: usize) -> bool {
    n >= 64
}

/// `[seq, 1, C]` with `seq > 1`, batch axis 1, channel on the last axis.
pub fn is_seq_first_rank3(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(0).unwrap_static() > 1
        && s.dim(1).unwrap_static() == 1
        && is_typical_channel(s.dim(2).unwrap_static())
}

/// `[1, seq, C]` BLC layout (batch 1, time on axis 1).
pub fn is_blc_rank3(s: &Shape) -> bool {
    s.rank() == 3
        && s.dim(0).unwrap_static() == 1
        && s.dim(1).unwrap_static() > 1
        && is_typical_channel(s.dim(2).unwrap_static())
}

/// When symbolic propagation yields BLC `[1, seq, C]` but the shape tensor evaluates
/// to seq-first `[seq, 1, C]`, prefer the evaluated layout (ONNX Expand broadcast).
pub fn prefer_seq_first_expand_target(evaluated: &Shape, from_meta: &Shape) -> Shape {
    if is_seq_first_rank3(evaluated) && is_blc_rank3(from_meta) {
        let ec = evaluated.dim(2).unwrap_static();
        let mc = from_meta.dim(2).unwrap_static();
        if ec == mc || mc == 1 {
            return evaluated.clone();
        }
    }
    from_meta.clone()
}

/// Reshape after bidirectional LSTM `Transpose` merges `[seq, *, batch, H]` → `[seq, batch, 2H]`.
pub fn bidir_lstm_merge_reshape_dims(in_shape: &Shape) -> Option<Vec<i64>> {
    if in_shape.rank() != 4 {
        return None;
    }
    let hidden = in_shape.dim(3).unwrap_static();
    if !is_typical_channel(hidden) {
        return None;
    }
    let merge = |seq: usize, batch: usize| -> Vec<i64> {
        vec![seq as i64, batch.max(1) as i64, (2 * hidden) as i64]
    };
    if in_shape.dim(1).unwrap_static() == 2 {
        return Some(merge(
            in_shape.dim(0).unwrap_static(),
            in_shape.dim(2).unwrap_static(),
        ));
    }
    if in_shape.dim(2).unwrap_static() == 2 {
        return Some(merge(
            in_shape.dim(0).unwrap_static(),
            in_shape.dim(1).unwrap_static(),
        ));
    }
    None
}

/// Build a static shape from evaluated i64 dims (negative → 1).
pub fn shape_from_i64_dims(dims: &[i64], dtype: rlx_ir::DType) -> Shape {
    let us: Vec<usize> = dims
        .iter()
        .map(|&d| if d < 0 { 1 } else { d as usize })
        .collect();
    Shape::new(&us, dtype)
}

/// ONNX Expand output rank: broadcast `input` with `target`, preserving trailing channels.
pub fn expand_output_dims(input: &[usize], target: &[usize]) -> Option<Vec<usize>> {
    if target.is_empty() {
        return Some(input.to_vec());
    }
    let rank = input.len().max(target.len());
    let mut out = vec![1usize; rank];
    for i in 0..rank {
        let in_d = if i + input.len() >= rank {
            input[i + input.len() - rank]
        } else {
            1
        };
        let tg_d = if i + target.len() >= rank {
            target[i + target.len() - rank]
        } else {
            1
        };
        out[i] = match (in_d, tg_d) {
            (a, b) if a == b => a,
            (1, b) => b,
            (a, 1) => a,
            _ if in_d == tg_d => in_d,
            _ => return None,
        };
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::DType;

    #[test]
    fn seq_first_vs_blc_detection() {
        let seq_first = Shape::new(&[8, 1, 512], DType::F32);
        let blc = Shape::new(&[1, 8, 512], DType::F32);
        assert!(is_seq_first_rank3(&seq_first));
        assert!(is_blc_rank3(&blc));
        assert!(!is_seq_first_rank3(&blc));
    }

    #[test]
    fn prefer_evaluated_expand_target() {
        let eval = Shape::new(&[8, 1, 128], DType::F32);
        let meta = Shape::new(&[1, 8, 128], DType::F32);
        let out = prefer_seq_first_expand_target(&eval, &meta);
        assert_eq!(out.dims(), eval.dims());
    }

    #[test]
    fn bidir_lstm_merge_from_onnx_layout() {
        let y = Shape::new(&[8, 2, 1, 256], DType::F32);
        assert_eq!(bidir_lstm_merge_reshape_dims(&y), Some(vec![8, 1, 512]));
        let y2 = Shape::new(&[8, 1, 2, 256], DType::F32);
        assert_eq!(bidir_lstm_merge_reshape_dims(&y2), Some(vec![8, 1, 512]));
    }

    #[test]
    fn expand_broadcast_style_to_seq() {
        assert_eq!(
            expand_output_dims(&[1, 128], &[8, 1, 1]),
            Some(vec![8, 1, 128])
        );
    }
}
