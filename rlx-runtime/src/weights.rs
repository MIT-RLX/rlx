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

//! Weight-loading abstraction.
//!
//! Native targets typically `mmap` a `.safetensors` file and read tensors
//! via byte-offset slices into the mapping. WASM has no `mmap`; weights
//! arrive as `Vec<u8>` from `fetch()` or `Response.arrayBuffer()`. Both
//! paths produce the same shape: a name → byte-slice lookup.
//!
//! `WeightLoader` is the contract. Concrete implementations live in
//! `rlx-models` (mmap-based) and here (`BytesWeightLoader` — works on
//! every target including WASM).

/// A name-keyed view of weight tensor bytes.
///
/// Implementations promise that the returned slice stays valid for the
/// lifetime of `&self`. On native, this is the mmap region; on WASM, it
/// is the in-memory `Vec<u8>` owned by the loader.
pub trait WeightLoader {
    /// Return the raw bytes for the tensor named `name`, or `None` if not
    /// present. Bytes are in the source file's storage order (typically
    /// row-major, dtype-native).
    fn tensor_bytes(&self, name: &str) -> Option<&[u8]>;

    /// All tensor names (for iteration / discovery). Order is
    /// implementation-defined but stable for a given loader instance.
    fn names(&self) -> Vec<String>;
}

/// Owned in-memory weight loader. The simplest, most portable variant —
/// works on every target including WASM.
///
/// Construct via `BytesWeightLoader::from_safetensors(bytes)` once
/// `rlx-models` integrates. For now the bare struct lets external callers
/// build their own name → bytes mapping.
pub struct BytesWeightLoader {
    /// `(name, start_offset, len)` triples into `data`.
    entries: Vec<(String, usize, usize)>,
    data: Vec<u8>,
}

impl BytesWeightLoader {
    /// Build a loader from a list of `(name, bytes)` pairs. Each tensor
    /// is appended into a single backing `Vec<u8>`; `tensor_bytes` returns
    /// a borrow into that vec.
    pub fn from_pairs(pairs: Vec<(String, Vec<u8>)>) -> Self {
        let total: usize = pairs.iter().map(|(_, b)| b.len()).sum();
        let mut data = Vec::with_capacity(total);
        let mut entries = Vec::with_capacity(pairs.len());
        for (name, bytes) in pairs {
            let start = data.len();
            let len = bytes.len();
            data.extend_from_slice(&bytes);
            entries.push((name, start, len));
        }
        Self { entries, data }
    }
}

impl WeightLoader for BytesWeightLoader {
    fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, off, len)| &self.data[*off..*off + *len])
    }

    fn names(&self) -> Vec<String> {
        self.entries.iter().map(|(n, _, _)| n.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let loader = BytesWeightLoader::from_pairs(vec![
            ("w".into(), vec![1, 2, 3, 4]),
            ("b".into(), vec![5, 6]),
        ]);
        assert_eq!(loader.tensor_bytes("w"), Some(&[1u8, 2, 3, 4][..]));
        assert_eq!(loader.tensor_bytes("b"), Some(&[5u8, 6][..]));
        assert_eq!(loader.tensor_bytes("missing"), None);
        assert_eq!(loader.names(), vec!["w".to_string(), "b".to_string()]);
    }
}
