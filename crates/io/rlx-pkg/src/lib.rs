// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! RLX package format (`.rlxp`) — flat mmap (default), ZIP, or directory.
//!
//! # Overview
//!
//! | Piece | Role |
//! |-------|------|
//! | [`Package`] | Open flat / ZIP / dir; mmap hot weights; inflate warm/cold |
//! | [`write_package`] | Emit a pack from MIR graph + [`PackedWeight`] rows |
//! | [`WriteOptions::include_graph`] | Optional executable MIR (default on); weight-only when false |
//! | [`StorageTier`] | Hot (mmap) / warm (`zstd_blocks`) / cold (zstd) |
//!
//! Spec: [`docs/rlxp.md`](../../../docs/rlxp.md).
//!
//! Bake exports via [`package_from_bake`] / `rlx-bake --format rlxp`.
//! ONNX import with an executable graph lives on `rlx-bake` (`--features onnx`).
//!
//! # Features
//!
//! Reading ([`Package`], every tier) is always built, in pure Rust — wasm32 too.
//!
//! - `write` (default) — [`write_package`], [`build`], auto-tiering, [`package_from_bake`]
//! - `convert` (default) — the `*_to_rlxp` importers (GGUF, MLX, DDUF, NeMo, PyTorch)
//! - `parallel` (default) — warm blocks decoded on rayon
//! - `zstd-native` — libzstd for the warm/cold codec (native; faster, real levels)
//! - `encrypt` — [`seal_bytes`] / [`unseal_bytes`] (`RLXSEAL1`)
//! - `remote` — [`RemoteFlat`] HTTP Range reader

#[cfg(feature = "write")]
mod auto_tier;
#[cfg(feature = "write")]
pub mod build;
mod flat;
#[cfg(feature = "write")]
mod from_bake;
#[cfg(feature = "convert")]
mod from_dduf;
#[cfg(feature = "convert")]
mod from_gguf;
#[cfg(feature = "convert")]
mod from_mlx;
#[cfg(feature = "convert")]
mod from_nemo;
#[cfg(feature = "convert")]
mod from_pt;
mod manifest;
mod package;
mod placement;
mod seals;
mod store_zip;
mod tier;
mod verify;
mod weights_index;
#[cfg(feature = "write")]
mod write;

#[cfg(feature = "remote")]
mod remote;

#[cfg(feature = "write")]
pub use auto_tier::{AutoTierOptions, apply_auto_tier};
pub use flat::{
    FLAG_BINCODE_TOC, FLAG_HYBRID, FLAT_CONTAINER_VERSION, FLAT_MAGIC, FlatHeader, FlatToc,
    is_flat_magic,
};
#[cfg(feature = "write")]
pub use from_bake::{BakeWeight, package_from_bake};
#[cfg(feature = "convert")]
pub use from_dduf::{DdufImportOptions, dduf_to_rlxp};
#[cfg(feature = "convert")]
pub use from_gguf::{GgufImportOptions, gguf_to_rlxp};
#[cfg(feature = "convert")]
pub use from_mlx::{MlxImportOptions, mlx_to_rlxp};
#[cfg(feature = "convert")]
pub use from_nemo::{NemoImportOptions, nemo_to_rlxp};
#[cfg(feature = "convert")]
pub use from_pt::{PtImportOptions, pt_to_rlxp};
pub use manifest::{
    COMPAT_VERSION, DistRef, FORMAT_NAME, FORMAT_VERSION, GRAPH_BINCODE, GRAPH_RLXMODEL, GraphRef,
    Manifest, SidecarRef, WeightsRef,
};
pub use package::{Bytes, MaterializeMode, MemberSource, Package, dtype_for_weight_scheme};
pub use placement::{ExpertPlacement, Placement, TensorShard};
pub use seals::{SEAL_MAGIC, is_sealed, seal_bytes, unseal_bytes};
pub use tier::{
    Codec, DEFAULT_WARM_BLOCK, StorageTier, checksum_hex, decode_zstd_block_at, decode_zstd_blocks,
    decode_zstd_blocks_parallel, encode_zstd_blocks,
};
pub use verify::{VerifyReport, verify_package};
pub use weights_index::{WeightEntry, WeightsIndex};
#[cfg(feature = "write")]
pub use write::{
    ContainerKind, PackedWeight, WriteOptions, graph_with_stripped_weights, infer_container,
    infer_zip_from_path, write_package, write_package_dir, write_package_flat, write_package_zip,
};

#[cfg(feature = "remote")]
pub use remote::{RemoteFlat, http_range};
