// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! `ndarray` interop (feature `ndarray`).
//!
//! Migrate in: `Tensor::from(array)` for any `ndarray::Array<f32, D>` (any
//! rank). Migrate out (with `eval`): `tensor.to_ndarray()` realizes the lazy
//! graph and hands back an owned `ArrayD<f32>`.

use ndarray::{Array, ArrayD, Dimension, IxDyn};

use crate::Tensor;

impl<D: Dimension> From<Array<f32, D>> for Tensor {
    fn from(a: Array<f32, D>) -> Self {
        Tensor::from(&a)
    }
}

impl<D: Dimension> From<&Array<f32, D>> for Tensor {
    fn from(a: &Array<f32, D>) -> Self {
        let dims = a.shape().to_vec();
        // `iter()` yields logical (row-major) order regardless of memory layout.
        let data: Vec<f32> = a.iter().copied().collect();
        Tensor::from_vec(data, dims)
    }
}

#[cfg(feature = "eval")]
impl Tensor {
    /// Realize (compile + run) and return an owned `ndarray::ArrayD<f32>`.
    pub fn to_ndarray(&self) -> ArrayD<f32> {
        let dims = self.dims();
        let data = self.to_vec();
        ArrayD::from_shape_vec(IxDyn(&dims), data)
            .expect("to_ndarray: realized data does not match shape")
    }
}
