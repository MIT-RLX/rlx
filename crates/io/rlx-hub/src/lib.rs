// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! # rlx-hub — shard-aware model download + verification
//!
//! Infrastructure for fetching (only) the part of a checkpoint a node needs and
//! checking it's intact. Two pieces:
//!
//! - [`index`] — parse `model.safetensors.index.json` and, given a pipeline
//!   stage's contiguous layer range, compute exactly which shard **files** hold
//!   its tensors ([`plan_layer_stages`]). Boundary shards land on adjacent
//!   stages so every node has *complete* layers. This is what lets three
//!   machines download ⅓ of a model each instead of the whole thing.
//! - [`download`] — resumable [`download::download_file`] / [`download::download_files`]
//!   (via `curl -C -`) plus [`download::verify_file`]: exact byte size (from the
//!   HF API) and a structural `.safetensors` header/data-length check that
//!   catches truncated or interrupted downloads.
//!
//! Model crates layer their specifics on top (which repo, which layer split per
//! node); this crate stays model-agnostic.
//!
//! ```no_run
//! use rlx_hub::{HfRepo, fetch_index, fetch_sizes, download_files, plan_layer_stages};
//! let repo = HfRepo::new("mlx-community/DeepSeek-V4-Flash-2bit-DQ");
//! let index = fetch_index(&repo)?;
//! let sizes = fetch_sizes(&repo)?;
//! // this node owns layers 18..35 (no embed/head)
//! let stage = &plan_layer_stages(&index, &[18..35], &[vec![]])[0];
//! let report = download_files(&repo, &stage.shards, "/models/ckpt".as_ref(), &sizes, |f, s| println!("{s}: {f}"));
//! assert!(report.failed.is_empty());
//! # Ok::<(), anyhow::Error>(())
//! ```

pub mod download;
pub mod index;

pub use download::{
    DownloadReport, HfRepo, curl_bytes, download_file, download_files, fetch_index, fetch_sizes,
    verify_file,
};
pub use index::{SafetensorsIndex, StageShards, even_layer_ranges, plan_layer_stages};
