// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

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
