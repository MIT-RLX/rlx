// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Turn a torch storage (`offset`, `size`, `stride` view over a flat
//! buffer) into a row-major contiguous `f32` tensor.

use anyhow::{Result, bail};

use crate::pickle::TensorMeta;

/// Materialize `meta` from its decoded storage values into a contiguous,
/// row-major `f32` vector of length `prod(shape)`.
///
/// `storage_f32` is the full decoded storage (all `numel` elements).
pub fn gather(meta: &TensorMeta, storage_f32: &[f32]) -> Result<Vec<f32>> {
    let numel: usize = meta.shape.iter().product();
    if meta.shape.len() != meta.stride.len() {
        bail!(
            "tensor {:?}: rank mismatch shape {:?} vs stride {:?}",
            meta.storage_key,
            meta.shape,
            meta.stride
        );
    }

    // Fast path: the view is already contiguous row-major from `offset`.
    if is_contiguous(&meta.shape, &meta.stride) {
        let start = meta.offset;
        let end = start + numel;
        if end > storage_f32.len() {
            bail!(
                "tensor {:?}: contiguous view [{start}..{end}) exceeds storage len {}",
                meta.storage_key,
                storage_f32.len()
            );
        }
        return Ok(storage_f32[start..end].to_vec());
    }

    // General strided gather.
    let mut out = Vec::with_capacity(numel);
    let rank = meta.shape.len();
    let mut idx = vec![0usize; rank];
    for _ in 0..numel {
        let mut src = meta.offset;
        for d in 0..rank {
            src += idx[d] * meta.stride[d];
        }
        let v = *storage_f32.get(src).ok_or_else(|| {
            anyhow::anyhow!(
                "tensor {:?}: strided index {src} out of range",
                meta.storage_key
            )
        })?;
        out.push(v);
        // increment the multi-dimensional counter (row-major / last dim fastest).
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < meta.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(out)
}

/// Is `(shape, stride)` a standard contiguous row-major layout?
fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    let mut expected = 1usize;
    for d in (0..shape.len()).rev() {
        // Dimensions of size 1 carry an arbitrary stride; ignore them.
        if shape[d] != 1 && stride[d] != expected {
            return false;
        }
        expected *= shape[d];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;

    fn meta(shape: Vec<usize>, stride: Vec<usize>, offset: usize) -> TensorMeta {
        TensorMeta {
            storage_key: "0".into(),
            dtype: DType::F32,
            shape,
            stride,
            offset,
        }
    }

    #[test]
    fn contiguous_slice() {
        let s: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let m = meta(vec![2, 3], vec![3, 1], 3);
        assert_eq!(gather(&m, &s).unwrap(), vec![3., 4., 5., 6., 7., 8.]);
    }

    #[test]
    fn transposed_view() {
        // 2x3 row-major storage viewed as its 3x2 transpose.
        let s: Vec<f32> = vec![0., 1., 2., 3., 4., 5.];
        let m = meta(vec![3, 2], vec![1, 3], 0);
        assert_eq!(gather(&m, &s).unwrap(), vec![0., 3., 1., 4., 2., 5.]);
    }
}
