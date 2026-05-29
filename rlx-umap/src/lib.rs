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

//! Parametric UMAP on RLX — k-NN building blocks + full [`Umap::fit`](umap::Umap::fit) API matching [fast-umap](https://github.com/eugenehp/fast-umap).
//!
//! - [`Umap::fit`] / [`fit_with_progress`](umap::Umap::fit_with_progress) — sparse cross-entropy training
//! - [`FittedUmap::transform`] — inference with training z-score stats
//! - [`FittedUmap::save`] / [`load`] — safetensors or GGUF (`.ruama` load only for legacy files)
//!
//! Call [`register`] once per process before Session execution.
//!
//! ## Quick start
//!
//! ```ignore
//! rlx_umap::register();
//!
//! use rlx_ir::{DType, Graph, Shape};
//! use rlx_umap::knn_indices_and_distances;
//!
//! let mut g = Graph::new("knn");
//! let pairwise = g.input("pairwise", Shape::new(&[64, 64], DType::F32));
//! let (idx, dist) = knn_indices_and_distances(&mut g, pairwise, 5);
//! g.set_outputs(vec![idx, dist]);
//! ```

pub mod knn;
pub mod knn_attrs;
pub mod pack;
pub mod pairwise;
pub mod parity;

#[cfg(feature = "cpu")]
pub mod ops;

#[cfg(feature = "graph")]
pub mod graph;

#[cfg(feature = "bench")]
pub mod session;

#[cfg(feature = "full")]
pub mod adam;
#[cfg(feature = "full")]
pub mod config;
#[cfg(feature = "full")]
pub mod data;
#[cfg(feature = "full")]
pub mod encoder;
#[cfg(feature = "full")]
pub mod fitted;
#[cfg(feature = "full")]
pub mod interrupt;
#[cfg(feature = "full")]
pub mod model;
#[cfg(feature = "full")]
pub mod model_io;
#[cfg(feature = "nn-descent")]
pub mod nn_descent;
#[cfg(feature = "pca")]
pub mod pca;
#[cfg(feature = "full")]
pub mod prelude;
#[cfg(feature = "full")]
pub mod serialize;
#[cfg(feature = "full")]
pub mod train;
#[cfg(feature = "full")]
pub mod training;
#[cfg(feature = "full")]
pub mod umap;
#[cfg(feature = "full")]
pub mod utils;
#[cfg(feature = "full")]
pub mod weights;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_kernels;

#[cfg(all(feature = "mlx", target_os = "macos"))]
mod mlx_kernels;

pub use knn::{knn_backward_pairwise, knn_forward_packed};
pub use knn_attrs::KnnAttrs;
pub use pack::unpack_knn_packed;
pub use pairwise::{cosine_pairwise_reference, euclidean_pairwise_reference};
pub use parity::{KnnParityReport, compare_knn, max_pairwise_error};

#[cfg(feature = "cpu")]
pub use ops::{UMAP_KNN, UMAP_KNN_BWD, register_umap_ops};

#[cfg(feature = "graph")]
pub use graph::{
    cosine_knn_graph, cosine_knn_packed_graph, knn_graph, knn_indices_and_distances,
    pairwise_cosine_graph, pairwise_euclidean_graph, split_knn_packed,
};

/// Register UMAP custom ops (IR + CPU). Alias of [`register_umap_ops`].
#[cfg(feature = "cpu")]
pub fn register() {
    register_umap_ops();
}

#[cfg(feature = "full")]
pub use config::UmapConfig;
#[cfg(feature = "full")]
pub use data::{load_csv, load_f64_matrix, load_synthetic, write_embedding_csv};
#[cfg(feature = "full")]
pub use fitted::FittedUmap;
#[cfg(feature = "full")]
pub use model_io::{
    EXT_GGUF, EXT_RUAMA, EXT_SAFETENSORS, MODEL_EXT, format_from_path, model_path,
    model_path_with_ext, weight_shapes,
};
#[cfg(feature = "full")]
pub use rlx_driver::Device;
#[cfg(feature = "full")]
pub use serialize::{
    LoadedModel, ModelMetadata, SaveBundle, load_model, load_weights, save_model, save_weights,
};
#[cfg(feature = "full")]
pub use train::EpochProgress;
#[cfg(feature = "full")]
pub use training::{FitOptions, TrainResult, fit, fit_with_progress, train_only};
#[cfg(feature = "full")]
pub use umap::Umap;
#[cfg(feature = "full")]
pub use utils::NormStats;
#[cfg(feature = "full")]
pub use weights::WeightStore;
