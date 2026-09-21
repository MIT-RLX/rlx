// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The **legacy** (pre-1.6) `torch.save` container.
//!
//! PyTorch switched to a ZIP container in 1.6 (mid-2020), but a large part of
//! the community model back-catalogue predates that and is still in daily use —
//! most of the ESRGAN-era super-resolution zoo, for one. Those files are not
//! ZIPs at all, so the modern reader rejects them outright.
//!
//! The layout is five pickles concatenated, then the raw tensor storages:
//!
//! ```text
//!   pickle  magic number      0x1950a86a20f9469cfc6c
//!   pickle  protocol version  1001
//!   pickle  sys_info          {protocol_version, little_endian, type_sizes}
//!   pickle  the object        state dict, tensors as persistent-id refs
//!   pickle  storage keys      ["0", "1", …] in write order
//!   ─────────────────────────────────────────────────────────────────
//!   per key: u64 element count, then count × element-size raw bytes
//! ```
//!
//! The object pickle is identical in shape to the modern `data.pkl` — same
//! `persistent_id` tuples, same `_rebuild_tensor_v2` — so the pickle VM needs
//! no changes; only the container walk is different.
//!
//! # Element size comes from the object, not the storage section
//!
//! The storage section records an element *count* and nothing else, so the
//! byte length cannot be computed without first knowing each storage's dtype —
//! which is only stated in the persistent ids inside the object pickle. That
//! ordering is why the object is parsed before the blobs are walked, and why a
//! storage referenced by no tensor cannot be sized (it is skipped, which is
//! safe because nothing can read it).
//!
//! Unlike the ZIP path, this reads the whole file: the storages are laid out
//! as one sequential stream with no index, so there is nothing to seek to.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::dtype::DType;
use crate::pickle::{TensorMeta, Value, unpickle_prefix};
use crate::torch::collect_state_dict;

/// A legacy checkpoint, fully read.
pub struct LegacyFile {
    pub tensors: BTreeMap<String, TensorMeta>,
    /// Storage key → raw little-endian bytes.
    pub storages: HashMap<String, Vec<u8>>,
}

impl std::fmt::Debug for LegacyFile {
    /// Elides the blobs: a legacy file's storages are the whole checkpoint, and
    /// a failed assertion that dumps 60 MB of weight bytes helps nobody.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes: usize = self.storages.values().map(Vec::len).sum();
        f.debug_struct("LegacyFile")
            .field("tensors", &self.tensors.len())
            .field("storages", &self.storages.len())
            .field("bytes", &bytes)
            .finish()
    }
}

/// Whether `head` looks like a legacy container rather than a ZIP.
///
/// A `torch.save` ZIP starts with the local-file-header magic `PK\x03\x04`; a
/// legacy file starts with the opening opcodes of a protocol-2 pickle.
pub fn looks_legacy(head: &[u8]) -> bool {
    !head.starts_with(b"PK\x03\x04")
}

/// Read a legacy `torch.save` file.
pub fn read(path: &Path) -> Result<LegacyFile> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut pos = 0usize;

    // magic, protocol version, sys_info — validated only loosely: a wrong
    // magic is worth reporting, but the other two carry nothing this needs.
    let (magic, n) = unpickle_prefix(&bytes).context("legacy torch file: magic number")?;
    pos += n;
    if let Value::Int(v) = magic {
        // 0x1950a86a20f9469cfc6c truncated into i64 by the pickle's LONG.
        if v == 0 {
            bail!("legacy torch file: magic number is zero");
        }
    }
    for what in ["protocol version", "sys_info"] {
        let (_, n) =
            unpickle_prefix(&bytes[pos..]).with_context(|| format!("legacy torch file: {what}"))?;
        pos += n;
    }

    let (obj, n) =
        unpickle_prefix(&bytes[pos..]).context("legacy torch file: the object pickle")?;
    pos += n;
    let tensors = collect_state_dict(&obj)?;

    let (keys, n) =
        unpickle_prefix(&bytes[pos..]).context("legacy torch file: the storage key list")?;
    pos += n;

    // Element size per storage, taken from the tensors that reference it.
    let mut dtype_of: HashMap<&str, DType> = HashMap::new();
    for t in tensors.values() {
        dtype_of.insert(t.storage_key.as_str(), t.dtype);
    }

    let keys: Vec<String> = match keys {
        Value::List(l) => l
            .borrow()
            .iter()
            .map(|v| v.as_str().map_err(anyhow::Error::from))
            .collect::<Result<_>>()?,
        Value::Tuple(t) => t
            .iter()
            .map(|v| v.as_str().map_err(anyhow::Error::from))
            .collect::<Result<_>>()?,
        other => bail!("legacy torch file: storage keys are not a list ({other:?})"),
    };

    let mut storages = HashMap::with_capacity(keys.len());
    for key in keys {
        if pos + 8 > bytes.len() {
            bail!("legacy torch file: storage {key:?} runs past the end of the file");
        }
        let numel = u64::from_le_bytes(bytes[pos..pos + 8].try_into().expect("8 bytes")) as usize;
        pos += 8;

        let Some(dtype) = dtype_of.get(key.as_str()).copied() else {
            // No tensor points at this storage, so its element size is unknown
            // and, equally, nothing can ever read it. Without a size the walk
            // cannot continue past it, so stop here rather than guess an
            // element width and desynchronize every storage after it.
            bail!(
                "legacy torch file: storage {key:?} is referenced by no tensor, so its \
                 element size is unknown and the remaining storages cannot be located"
            );
        };
        let len = numel * dtype.size();
        if pos + len > bytes.len() {
            bail!(
                "legacy torch file: storage {key:?} wants {len} bytes but only {} remain",
                bytes.len() - pos
            );
        }
        storages.insert(key, bytes[pos..pos + len].to_vec());
        pos += len;
    }

    Ok(LegacyFile { tensors, storages })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal protocol-2 pickle writer, enough to synthesize a legacy file.
    ///
    /// Writing the container by hand is the only way to test the *walk* without
    /// shipping a binary fixture or depending on a download: the reader's real
    /// work is locating storages, and that only happens on bytes.
    #[derive(Default)]
    struct Pk(Vec<u8>);

    impl Pk {
        fn new() -> Self {
            Pk(vec![0x80, 0x02])
        }
        fn raw(mut self, b: &[u8]) -> Self {
            self.0.extend_from_slice(b);
            self
        }
        fn long1(self, v: u128, n: usize) -> Self {
            let mut b = vec![0x8a, n as u8];
            b.extend_from_slice(&v.to_le_bytes()[..n]);
            self.raw(&b)
        }
        fn int(self, v: i32) -> Self {
            let mut b = vec![b'J'];
            b.extend_from_slice(&v.to_le_bytes());
            self.raw(&b)
        }
        fn u8int(self, v: u8) -> Self {
            self.raw(&[b'K', v])
        }
        fn str(self, s: &str) -> Self {
            let mut b = vec![b'X'];
            b.extend_from_slice(&(s.len() as u32).to_le_bytes());
            b.extend_from_slice(s.as_bytes());
            self.raw(&b)
        }
        fn global(self, m: &str, n: &str) -> Self {
            self.raw(format!("c{m}\n{n}\n").as_bytes())
        }
        fn stop(mut self) -> Vec<u8> {
            self.0.push(b'.');
            self.0
        }
    }

    /// The `("storage", FloatStorage, key, "cpu", numel)` persistent id, then
    /// the `_rebuild_tensor_v2` call that turns it into a tensor.
    fn tensor(pk: Pk, key: &str, numel: u8, dims: &[u8]) -> Pk {
        let mut pk = pk
            .global("torch._utils", "_rebuild_tensor_v2")
            .raw(b"((")
            .str("storage")
            .global("torch", "FloatStorage")
            .str(key)
            .str("cpu")
            .u8int(numel)
            .raw(b"tQ")
            .u8int(0); // storage_offset
        pk = pk.raw(b"(");
        for &d in dims {
            pk = pk.u8int(d);
        }
        pk = pk.raw(b"t(");
        // Contiguous strides, innermost first.
        let mut stride = 1u8;
        let mut strides = vec![];
        for &d in dims.iter().rev() {
            strides.push(stride);
            stride *= d;
        }
        for &s in strides.iter().rev() {
            pk = pk.u8int(s);
        }
        pk.raw(b"t\x89}tR") // strides, requires_grad=False, hooks, TUPLE, REDUCE
    }

    /// Assemble the five pickles plus the storage section.
    fn legacy_file(tensors: &[(&str, &str, u8, &[u8])], keys: &[&str]) -> Vec<u8> {
        let mut out = Pk::new().long1(0x1950_a86a_20f9_469c_fc6c, 10).stop();
        out.extend(Pk::new().int(1001).stop());
        out.extend(Pk::new().raw(b"}").stop());

        let mut obj = Pk::new().raw(b"}(");
        for (name, key, numel, dims) in tensors {
            obj = tensor(obj.str(name), key, *numel, dims);
        }
        out.extend(obj.raw(b"u").stop());

        let mut kl = Pk::new().raw(b"](");
        for k in keys {
            kl = kl.str(k);
        }
        out.extend(kl.raw(b"e").stop());

        for (_, _, numel, _) in tensors {
            out.extend_from_slice(&(*numel as u64).to_le_bytes());
            for i in 0..*numel {
                out.extend_from_slice(&(i as f32).to_le_bytes());
            }
        }
        out
    }

    fn write(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("rlx-legacy-{name}-{}.pt", std::process::id()));
        std::fs::write(&p, bytes).expect("temp file");
        p
    }

    #[test]
    fn a_zip_header_is_not_legacy() {
        assert!(!looks_legacy(b"PK\x03\x04rest"));
        // Protocol-2 pickles open with PROTO 2.
        assert!(looks_legacy(b"\x80\x02\x8a\x0a"));
    }

    #[test]
    fn reads_a_synthetic_legacy_container() {
        let bytes = legacy_file(&[("w", "0", 4, &[2, 2]), ("b", "1", 2, &[2])], &["0", "1"]);
        assert!(looks_legacy(&bytes));
        let p = write("ok", &bytes);
        let f = read(&p).expect("legacy read");

        assert_eq!(f.tensors.len(), 2);
        let w = &f.tensors["w"];
        assert_eq!(w.shape, vec![2, 2]);
        assert_eq!(w.dtype, DType::F32);
        assert_eq!(w.storage_key, "0");
        assert_eq!(f.tensors["b"].shape, vec![2]);

        // Four f32 for `w`, two for `b` — the walk must land on the right
        // boundary, which it can only do by sizing storage 0 from its dtype.
        assert_eq!(f.storages["0"].len(), 16);
        assert_eq!(f.storages["1"].len(), 8);
        assert_eq!(&f.storages["1"][..4], &0f32.to_le_bytes());
        assert_eq!(&f.storages["1"][4..], &1f32.to_le_bytes());
        let _ = std::fs::remove_file(p);
    }

    /// An unreferenced storage cannot be sized, so every storage after it would
    /// be read from the wrong offset. That must fail loudly rather than return
    /// plausible garbage.
    #[test]
    fn an_unreferenced_storage_is_an_error() {
        let mut bytes = legacy_file(&[("w", "0", 4, &[2, 2])], &["0", "ghost"]);
        // The ghost's own length header is present and its payload is intact —
        // only its element *width* is unknowable, so this isolates that case
        // from a merely truncated file.
        bytes.extend_from_slice(&4u64.to_le_bytes());
        bytes.extend_from_slice(&[0u8; 16]);
        let p = write("ghost", &bytes);
        let e = read(&p).expect_err("unreferenced storage must fail");
        assert!(
            format!("{e:#}").contains("referenced by no tensor"),
            "unexpected error: {e:#}"
        );
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn a_truncated_storage_is_an_error() {
        let mut bytes = legacy_file(&[("w", "0", 4, &[2, 2])], &["0"]);
        bytes.truncate(bytes.len() - 6);
        let p = write("short", &bytes);
        let e = read(&p).expect_err("truncated file must fail");
        assert!(
            format!("{e:#}").contains("but only"),
            "unexpected error: {e:#}"
        );
        let _ = std::fs::remove_file(p);
    }
}
