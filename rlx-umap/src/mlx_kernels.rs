// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! MLX registration for `umap.knn`.
//!
//! MLX builds the full graph inside `mlx::compile`; host `eval` / `to_bytes` inside
//! `MlxKernel::execute` is not allowed there. Use [`crate::session::cosine_knn_mlx`]
//! (pairwise on MLX, k-NN on CPU) for end-to-end parity.

#![cfg(all(feature = "mlx", target_os = "macos"))]

use std::sync::Arc;

use rlx_ir::Shape;
use rlx_mlx::MlxError;
use rlx_mlx::array::Array;
use rlx_mlx::op_registry::{MlxKernel, register_mlx_kernel};

use crate::ops::UMAP_KNN;

#[derive(Debug)]
struct KnnForwardMlx;

impl MlxKernel for KnnForwardMlx {
    fn name(&self) -> &str {
        UMAP_KNN
    }

    fn execute(
        &self,
        _inputs: &[&Array],
        _output_shape: &Shape,
        _attrs: &[u8],
    ) -> Result<Array, MlxError> {
        Err(MlxError(
            "umap.knn on MLX: use rlx_umap::session::cosine_knn_mlx \
             (MLX pairwise + CPU k-NN) — host k-NN cannot run inside mlx::compile"
                .into(),
        ))
    }
}

pub fn register_mlx_kernels() {
    register_mlx_kernel(Arc::new(KnnForwardMlx));
}
