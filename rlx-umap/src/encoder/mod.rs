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

pub mod knn;
pub mod loss;
pub mod mlp;
#[cfg(feature = "pca")]
pub mod pca_warmstart;

pub use knn::{
    build_knn_edges, knn_edge_match_rate, knn_index_match_rate, knn_indices_cpu,
    knn_indices_device_fused, knn_indices_from_pairwise, pairwise_matrix_cpu,
};
pub use loss::{UmapTrainGraph, build_train_graph};
pub use mlp::{ModelSpec, ParamSlot, build_forward_graph, init_model_weights};
