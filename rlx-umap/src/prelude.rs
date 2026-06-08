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

//! Common re-exports for parametric UMAP on RLX.

pub use crate::config::{
    GraphParams, LossReduction, ManifoldParams, Metric, OptimizationParams, UmapConfig, fit_ab,
};
pub use crate::data::{load_csv, load_synthetic, write_embedding_csv};
pub use crate::fitted::FittedUmap;
pub use crate::model_io::MODEL_EXT;
pub use crate::serialize::{
    LoadedModel, ModelMetadata, SaveBundle, load_model, load_weights, model_path, save_model,
    save_weights,
};
pub use crate::train::EpochProgress;
pub use crate::training::{FitOptions, fit, fit_with_progress, train_only};
pub use crate::umap::Umap;
pub use crate::utils::{NormStats, generate_test_data};
pub use crate::weights::WeightStore;
pub use crate::{
    compare_knn, cosine_knn_graph, cosine_pairwise_reference, euclidean_pairwise_reference,
    knn_forward_packed, knn_graph, max_pairwise_error, pairwise_cosine_graph,
    pairwise_euclidean_graph, register,
};
