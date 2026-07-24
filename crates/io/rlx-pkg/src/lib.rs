// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

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
//! - `encrypt` — [`seal_bytes`] / [`unseal_bytes`] (`RLXSEAL1`)
//! - `remote` — [`RemoteFlat`] HTTP Range reader

mod auto_tier;
mod flat;
mod from_bake;
mod from_gguf;
mod manifest;
mod package;
mod placement;
mod seals;
mod store_zip;
mod tier;
mod verify;
mod weights_index;
mod write;

#[cfg(feature = "remote")]
mod remote;

pub use auto_tier::{AutoTierOptions, apply_auto_tier};
pub use flat::{
    FLAT_CONTAINER_VERSION, FLAT_MAGIC, FLAG_BINCODE_TOC, FLAG_HYBRID, FlatHeader, FlatToc,
    is_flat_magic,
};
pub use from_bake::{BakeWeight, package_from_bake};
pub use from_gguf::{GgufImportOptions, gguf_to_rlxp};
pub use manifest::{
    COMPAT_VERSION, FORMAT_NAME, FORMAT_VERSION, DistRef, GraphRef, Manifest, SidecarRef,
    WeightsRef,
};
pub use package::{MaterializeMode, MemberSource, Package};
pub use placement::{ExpertPlacement, Placement, TensorShard};
pub use seals::{SEAL_MAGIC, is_sealed, seal_bytes, unseal_bytes};
pub use tier::{
    Codec, DEFAULT_WARM_BLOCK, StorageTier, checksum_hex, decode_zstd_block_at, decode_zstd_blocks,
    decode_zstd_blocks_parallel, encode_zstd_blocks,
};
pub use verify::{VerifyReport, verify_package};
pub use weights_index::{WeightEntry, WeightsIndex};
pub use write::{
    ContainerKind, PackedWeight, WriteOptions, graph_with_stripped_weights, infer_container,
    infer_zip_from_path, write_package, write_package_dir, write_package_flat, write_package_zip,
};

#[cfg(feature = "remote")]
pub use remote::{RemoteFlat, http_range};
