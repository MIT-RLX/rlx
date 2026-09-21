// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Native reader for PyTorch **`torch.save`** checkpoints — `.pt`, `.pth`,
//! `pytorch_model.bin`, and the `model_weights.ckpt` embedded in a `.nemo`.
//!
//! A `torch.save` file (PyTorch ≥ 1.6, the default since 2020) is a ZIP
//! holding a pickled object graph in `data.pkl` plus one raw storage blob
//! per tensor under `data/<key>`. This crate indexes that container without
//! decompressing or copying the weight blob, unpickles the state dict into a
//! flat name → [`TensorMeta`] table, and materializes individual tensors on
//! demand as contiguous, row-major `f32` regardless of their on-disk dtype
//! (fp32 / fp16 / bf16 / int).
//!
//! The **legacy** pre-1.6 container — five concatenated pickles followed by a
//! flat storage stream, with no ZIP around it — is read too, dispatched on the
//! file header. Much of the community back-catalogue predates the 2020 switch
//! and is still in daily use, so rejecting it would lock out a large share of
//! real checkpoints for a format difference the caller never chose.
//!
//! There is no libtorch, no Python, and no protobuf here — only a ZIP reader,
//! a pickle VM restricted to what `torch.save` emits, and dtype decoders.
//!
//! ```no_run
//! use rlx_torch_ckpt::PtModel;
//! let m = PtModel::open(std::path::Path::new("4x_model.pth"))?;
//! for name in m.names() {
//!     let t = m.tensor(&name)?; // -> PtTensor (f32)
//!     println!("{name}: {:?}", t.shape);
//! }
//! # anyhow::Ok(())
//! ```
//!
//! # Relationship to the other torch crates
//!
//! * `rlx-torch-ckpt` (this crate) reads **weights** — a saved `state_dict`.
//! * `rlx-torch-import` reads **programs** — a `torch.export`ed graph, which
//!   it lowers to RLX HIR.
//! * `rlx-nemo` wraps this crate with the `.nemo` tar + YAML config layer.
//!
//! # Archive helpers
//!
//! [`archive`] is public because `.nemo` (tar-wrapping-ZIP) and other
//! container formats need the same seek/list/read primitives; it is a thin
//! layer, not a general-purpose archive library.

pub mod archive;
mod dtype;
mod legacy;
mod pickle;
mod pt;
pub(crate) mod storage;
mod torch;

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

pub use archive::{Seekable, TarMember, ZipEntry};
pub use dtype::DType;
pub use pickle::TensorMeta;
pub use pt::{PtModel, PtTensor};

use archive::read_zip_entry;

/// Index a `torch.save` ZIP: unpickle its `data.pkl` into a flat tensor
/// table and map each storage key to its `data/<key>` ZIP entry.
///
/// `entries` must be the ZIP's entry list with **absolute** file offsets, so
/// the same routine serves a bare `.pth` (ZIP at offset 0) and a checkpoint
/// nested inside another container such as a `.nemo` tar.
pub fn index_torch_zip(
    file: &mut File,
    entries: &[ZipEntry],
) -> Result<(BTreeMap<String, TensorMeta>, HashMap<String, ZipEntry>)> {
    // Locate `<archive>/data.pkl` and the `<archive>/data/<key>` storages.
    let pkl_entry = entries
        .iter()
        .find(|e| e.name.ends_with("data.pkl"))
        .ok_or_else(|| anyhow!("no data.pkl in checkpoint zip"))?;
    let archive_prefix = pkl_entry
        .name
        .strip_suffix("data.pkl")
        .unwrap_or("")
        .to_string();
    let data_prefix = format!("{archive_prefix}data/");

    let mut storages = HashMap::new();
    for e in entries {
        if let Some(key) = e.name.strip_prefix(&data_prefix) {
            if !key.is_empty() {
                storages.insert(key.to_string(), e.clone());
            }
        }
    }

    let pkl_bytes = read_zip_entry(file, pkl_entry)?;
    let root = pickle::unpickle(&pkl_bytes).context("unpickling data.pkl")?;
    let tensors = torch::collect_state_dict(&root)?;
    Ok((tensors, storages))
}

/// Materialize one tensor's storage view as a contiguous, row-major `f32`
/// vector, given the container path and the storage-key → ZIP-entry map.
pub fn read_torch_tensor(
    path: &Path,
    meta: &TensorMeta,
    storages: &HashMap<String, ZipEntry>,
) -> Result<Vec<f32>> {
    let entry = storages
        .get(&meta.storage_key)
        .ok_or_else(|| anyhow!("missing storage {:?}", meta.storage_key))?;

    let expected = meta.dtype.size() as u64;
    if entry.size % expected != 0 {
        bail!(
            "storage {:?}: {} bytes not a multiple of dtype width {}",
            meta.storage_key,
            entry.size,
            expected
        );
    }

    let mut file = File::open(path)?;
    let raw = read_zip_entry(&mut file, entry)?;
    let storage_f32 = meta.dtype.decode_f32(&raw);
    storage::gather(meta, &storage_f32)
}
