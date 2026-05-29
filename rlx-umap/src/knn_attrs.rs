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

//! Attribute blob for `umap.knn` / `umap.knn_backward` (`k` neighbours per row).

/// Per-instance knobs for k-NN custom ops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnnAttrs {
    /// Number of nearest neighbours per row (must be `< n`).
    pub k: u32,
}

impl KnnAttrs {
    pub const ENCODE_LEN: usize = 4;

    pub fn encode(self) -> Vec<u8> {
        self.k.to_le_bytes().to_vec()
    }

    pub fn decode(attrs: &[u8]) -> Result<Self, String> {
        let bytes: [u8; 4] = attrs
            .get(..Self::ENCODE_LEN)
            .and_then(|s| s.try_into().ok())
            .ok_or_else(|| {
                format!(
                    "knn attrs: expected {} bytes, got {}",
                    Self::ENCODE_LEN,
                    attrs.len()
                )
            })?;
        Ok(Self {
            k: u32::from_le_bytes(bytes),
        })
    }
}
