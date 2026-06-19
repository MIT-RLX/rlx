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

//! Standalone loader for a plain PyTorch `torch.save` checkpoint — a
//! `.pt` / `.pth` / `pytorch_model.bin` file.
//!
//! A `.nemo` wraps the very same checkpoint inside a tar alongside a YAML
//! config; a `.pt` *is* that checkpoint ZIP, so we parse it directly from
//! offset 0 and reuse the shared [`crate::index_torch_zip`] /
//! [`crate::read_torch_tensor`] machinery. Only the modern ZIP format
//! (PyTorch ≥ 1.6, the default since 2020) is supported — the legacy
//! non-ZIP pickle format is rejected with a clear error.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result, anyhow};

use crate::archive::{Seekable, ZipEntry, list_zip, prepare_seekable};
use crate::dtype::DType;
use crate::pickle::TensorMeta;
use crate::{index_torch_zip, read_torch_tensor};

/// A tensor materialized from a `.pt` checkpoint as contiguous f32.
#[derive(Debug, Clone)]
pub struct PtTensor {
    pub name: String,
    pub shape: Vec<usize>,
    /// The on-disk dtype before conversion to f32 (for reporting/quant).
    pub dtype: DType,
    pub data: Vec<f32>,
}

/// An opened PyTorch checkpoint: a lazily-readable tensor index over the
/// `torch.save` ZIP. Tensor storages are pulled on demand and materialized
/// as contiguous, row-major f32 regardless of their on-disk dtype.
///
/// ```no_run
/// use rlx_nemo::PtModel;
/// let m = PtModel::open(std::path::Path::new("pytorch_model.bin"))?;
/// for name in m.names() {
///     let t = m.tensor(&name)?; // -> PtTensor (f32)
///     println!("{name}: {:?}", t.shape);
/// }
/// # anyhow::Ok(())
/// ```
pub struct PtModel {
    /// The seekable source (a temp decompressed copy only in the unlikely
    /// event the `.pt` is gzip-wrapped; kept alive so tensor reads stay
    /// valid).
    source: Seekable,
    /// Param name → tensor view metadata (from the pickle).
    tensors: BTreeMap<String, TensorMeta>,
    /// Storage key → its (absolute-offset) ZIP entry.
    storages: HashMap<String, ZipEntry>,
}

impl PtModel {
    /// Open and index a `.pt` / `.pth` / `pytorch_model.bin` file.
    pub fn open(path: &Path) -> Result<Self> {
        let source =
            prepare_seekable(path).with_context(|| format!("preparing {}", path.display()))?;
        let read_path = source.path().to_path_buf();
        let mut file =
            File::open(&read_path).with_context(|| format!("opening {}", read_path.display()))?;
        let len = file.metadata()?.len();

        // The whole file is the torch.save ZIP (offset 0, full length).
        let entries = list_zip(&mut file, 0, len).with_context(|| {
            format!(
                "reading {} as a torch.save zip (legacy non-zip .pt files are not supported)",
                path.display()
            )
        })?;
        let (tensors, storages) = index_torch_zip(&mut file, &entries)
            .with_context(|| format!("indexing {}", path.display()))?;

        Ok(Self {
            source,
            tensors,
            storages,
        })
    }

    /// All tensor names, sorted.
    pub fn names(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    /// Number of tensors in the checkpoint.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Shape of a tensor without reading its data.
    pub fn shape_of(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// On-disk dtype of a tensor without reading its data.
    pub fn dtype_of(&self, name: &str) -> Option<DType> {
        self.tensors.get(name).map(|t| t.dtype)
    }

    /// Read one tensor and materialize it as contiguous f32.
    pub fn tensor(&self, name: &str) -> Result<PtTensor> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("no tensor named {name:?}"))?;
        let data = read_torch_tensor(self.source.path(), meta, &self.storages)
            .with_context(|| format!("reading tensor {name:?}"))?;
        Ok(PtTensor {
            name: name.to_string(),
            shape: meta.shape.clone(),
            dtype: meta.dtype,
            data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal STORED (uncompressed) ZIP — the layout `torch.save`
    /// emits. CRCs are left zero; our reader keys off the central directory
    /// + local headers and never validates them.
    fn zip_stored(files: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        let mut offsets = Vec::new();

        for (name, data) in files {
            offsets.push(out.len() as u32);
            let nb = name.as_bytes();
            out.extend_from_slice(&[0x50, 0x4b, 0x03, 0x04]); // local file header sig
            out.extend_from_slice(&[20, 0]); // version needed
            out.extend_from_slice(&[0, 0]); // flags
            out.extend_from_slice(&[0, 0]); // method = STORED
            out.extend_from_slice(&[0, 0, 0, 0]); // mod time/date
            out.extend_from_slice(&[0, 0, 0, 0]); // crc32
            out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // comp size
            out.extend_from_slice(&(data.len() as u32).to_le_bytes()); // uncomp size
            out.extend_from_slice(&(nb.len() as u16).to_le_bytes()); // name len
            out.extend_from_slice(&[0, 0]); // extra len
            out.extend_from_slice(nb);
            out.extend_from_slice(data);
        }

        let cd_offset = out.len() as u32;
        for ((name, data), &off) in files.iter().zip(&offsets) {
            let nb = name.as_bytes();
            central.extend_from_slice(&[0x50, 0x4b, 0x01, 0x02]); // central dir header sig
            central.extend_from_slice(&[20, 0]); // version made by
            central.extend_from_slice(&[20, 0]); // version needed
            central.extend_from_slice(&[0, 0]); // flags
            central.extend_from_slice(&[0, 0]); // method = STORED
            central.extend_from_slice(&[0, 0, 0, 0]); // mod time/date
            central.extend_from_slice(&[0, 0, 0, 0]); // crc32
            central.extend_from_slice(&(data.len() as u32).to_le_bytes()); // comp size
            central.extend_from_slice(&(data.len() as u32).to_le_bytes()); // uncomp size
            central.extend_from_slice(&(nb.len() as u16).to_le_bytes()); // name len
            central.extend_from_slice(&[0, 0]); // extra len
            central.extend_from_slice(&[0, 0]); // comment len
            central.extend_from_slice(&[0, 0]); // disk number start
            central.extend_from_slice(&[0, 0]); // internal attrs
            central.extend_from_slice(&[0, 0, 0, 0]); // external attrs
            central.extend_from_slice(&off.to_le_bytes()); // local header offset
            central.extend_from_slice(nb);
        }
        let cd_size = central.len() as u32;
        out.extend_from_slice(&central);

        // End Of Central Directory.
        out.extend_from_slice(&[0x50, 0x4b, 0x05, 0x06]);
        out.extend_from_slice(&[0, 0]); // disk number
        out.extend_from_slice(&[0, 0]); // disk with central dir
        let n = files.len() as u16;
        out.extend_from_slice(&n.to_le_bytes()); // entries this disk
        out.extend_from_slice(&n.to_le_bytes()); // total entries
        out.extend_from_slice(&cd_size.to_le_bytes());
        out.extend_from_slice(&cd_offset.to_le_bytes());
        out.extend_from_slice(&[0, 0]); // comment len
        out
    }

    fn binunicode(s: &str) -> Vec<u8> {
        let mut v = vec![b'X'];
        v.extend_from_slice(&(s.len() as u32).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
        v
    }

    /// Hand-assemble the `data.pkl` for a state dict `{"w": tensor[2,2]}`
    /// stored as `FloatStorage` key "0". Mirrors what `torch.save` emits,
    /// restricted to the opcode subset the VM understands.
    fn pickle_one_tensor() -> Vec<u8> {
        let mut p: Vec<u8> = vec![0x80, 0x02]; // PROTO 2
        p.push(b'}'); // EMPTY_DICT
        p.push(b'('); // MARK (for SETITEMS)
        p.extend(binunicode("w")); // key

        // value = _rebuild_tensor_v2(storage, 0, (2,2), (2,1), False, None)
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n"); // GLOBAL func
        p.push(b'('); // MARK (args)

        // storage = persistent_load(("storage", torch.FloatStorage, "0", "cpu", 4))
        p.push(b'('); // MARK (persid tuple)
        p.extend(binunicode("storage"));
        p.extend_from_slice(b"ctorch\nFloatStorage\n"); // GLOBAL storage type
        p.extend(binunicode("0")); // key
        p.extend(binunicode("cpu")); // location
        p.extend_from_slice(&[b'K', 4]); // numel
        p.push(b't'); // TUPLE -> 5-tuple
        p.push(b'Q'); // BINPERSID -> Storage

        p.extend_from_slice(&[b'K', 0]); // storage_offset

        p.push(b'('); // size tuple (2, 2)
        p.extend_from_slice(&[b'K', 2, b'K', 2]);
        p.push(b't');

        p.push(b'('); // stride tuple (2, 1)
        p.extend_from_slice(&[b'K', 2, b'K', 1]);
        p.push(b't');

        p.push(0x89); // requires_grad = False
        p.push(b'N'); // backward_hooks = None

        p.push(b't'); // TUPLE -> args
        p.push(b'R'); // REDUCE -> Tensor

        p.push(b'u'); // SETITEMS: {"w": tensor}
        p.push(b'.'); // STOP
        p
    }

    fn write_pt(values: &[f32]) -> tempfile::NamedTempFile {
        let mut storage = Vec::new();
        for &v in values {
            storage.extend_from_slice(&v.to_le_bytes());
        }
        let zip = zip_stored(&[
            ("archive/data.pkl", pickle_one_tensor()),
            ("archive/data/0", storage),
        ]);
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(&zip).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn loads_synthetic_pt() {
        let vals = vec![1.0f32, 2.0, 3.0, 4.0];
        let f = write_pt(&vals);
        let m = PtModel::open(f.path()).unwrap();

        assert_eq!(m.names(), vec!["w".to_string()]);
        assert_eq!(m.len(), 1);
        assert!(!m.is_empty());
        assert_eq!(m.shape_of("w"), Some(&[2usize, 2][..]));
        assert_eq!(m.dtype_of("w"), Some(DType::F32));

        let t = m.tensor("w").unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.data, vals);
    }

    #[test]
    fn missing_tensor_errors() {
        let f = write_pt(&[1.0, 2.0, 3.0, 4.0]);
        let m = PtModel::open(f.path()).unwrap();
        assert!(m.tensor("nope").is_err());
    }
}
