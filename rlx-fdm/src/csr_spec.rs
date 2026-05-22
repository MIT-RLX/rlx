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

//! Fixed-topology CSR stiffness assembly `values = f(q)` for MIR (`feature rlx-sparse`).

use crate::sparse::pattern_fast;
use crate::sparse_fast::SparseStiffnessFast;
use crate::structure::Structure;

/// Off-diagonal NZ contribution and diagonal overwrite maps for one network topology.
#[derive(Clone, Debug)]
pub struct CsrAssemblySpec {
    pub num_edges: usize,
    pub nnz: usize,
    /// Per-nnz off-diagonal terms before diagonal overwrite.
    pub nz_terms: Vec<Vec<(u32, f64)>>,
    /// `(nz_index, incident edge indices)` for diagonal entries.
    pub diag_overrides: Vec<(u32, Vec<u32>)>,
}

impl CsrAssemblySpec {
    pub fn from_structure(structure: &Structure) -> Self {
        let mut spec = Self::from_pattern(&pattern_fast(structure));
        spec.num_edges = structure.num_edges;
        spec
    }

    pub fn from_pattern(pat: &SparseStiffnessFast) -> Self {
        let nnz = pat.col_idx.len();
        let mut nz_terms = Vec::with_capacity(nnz);
        for terms in &pat.nz_terms {
            nz_terms.push(
                terms
                    .iter()
                    .map(|t| (t.edge as u32, t.coef as f64))
                    .collect(),
            );
        }
        let mut diag_overrides = Vec::new();
        for (row, edges) in pat.diag_edges.iter().enumerate() {
            let rs = pat.row_ptr[row] as usize;
            let re = pat.row_ptr[row + 1] as usize;
            for k in rs..re {
                if pat.col_idx[k] as usize == row {
                    diag_overrides.push((
                        k as u32,
                        edges.iter().map(|&e| e as u32).collect(),
                    ));
                    break;
                }
            }
        }
        Self {
            num_edges: 0,
            nnz,
            nz_terms,
            diag_overrides,
        }
    }

    /// Host forward: `values = assemble(q)` (matches [`SparseStiffnessFast::assemble`]).
    pub fn assemble(&self, q: &[f64]) -> Vec<f64> {
        let mut values = vec![0.0; self.nnz];
        for (k, terms) in self.nz_terms.iter().enumerate() {
            let mut v = 0.0;
            for &(e, coef) in terms {
                v += -q[e as usize] * coef;
            }
            values[k] = v;
        }
        for &(k, ref edges) in &self.diag_overrides {
            let mut d = 0.0;
            for &e in edges {
                d += -q[e as usize];
            }
            values[k as usize] = d;
        }
        values
    }

    /// Host VJP: `d_q = ∂(sum_k d_values[k] * values[k]) / ∂q`.
    pub fn vjp(&self, d_values: &[f64], q: &[f64]) -> Vec<f64> {
        let _ = q;
        let mut dq = vec![0.0; self.num_edges];
        let mut diag_k = std::collections::HashSet::new();
        for &(k, _) in &self.diag_overrides {
            diag_k.insert(k as usize);
        }
        for (k, terms) in self.nz_terms.iter().enumerate() {
            if diag_k.contains(&k) {
                continue;
            }
            let gk = d_values[k];
            if gk == 0.0 {
                continue;
            }
            for &(e, coef) in terms {
                dq[e as usize] += gk * (-coef);
            }
        }
        for &(k, ref edges) in &self.diag_overrides {
            let gk = d_values[k as usize];
            if gk == 0.0 {
                continue;
            }
            for &e in edges {
                dq[e as usize] += gk * (-1.0);
            }
        }
        dq
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"FDM1");
        out.extend_from_slice(&(self.num_edges as u32).to_le_bytes());
        out.extend_from_slice(&(self.nnz as u32).to_le_bytes());
        for terms in &self.nz_terms {
            out.extend_from_slice(&(terms.len() as u16).to_le_bytes());
            for &(e, coef) in terms {
                out.extend_from_slice(&e.to_le_bytes());
                out.extend_from_slice(&coef.to_le_bytes());
            }
        }
        out.extend_from_slice(&(self.diag_overrides.len() as u32).to_le_bytes());
        for &(k, ref edges) in &self.diag_overrides {
            out.extend_from_slice(&k.to_le_bytes());
            out.extend_from_slice(&(edges.len() as u16).to_le_bytes());
            for &e in edges {
                out.extend_from_slice(&e.to_le_bytes());
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 12 || &bytes[0..4] != b"FDM1" {
            return Err("csr spec: expected FDM1 header".into());
        }
        let num_edges = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let nnz = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut off = 12usize;
        let mut nz_terms = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            if off + 2 > bytes.len() {
                return Err("csr spec: truncated nz_terms".into());
            }
            let n = u16::from_le_bytes(bytes[off..off + 2].try_into().unwrap()) as usize;
            off += 2;
            let mut terms = Vec::with_capacity(n);
            for _ in 0..n {
                if off + 12 > bytes.len() {
                    return Err("csr spec: truncated term".into());
                }
                let e = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
                let coef = f64::from_le_bytes(bytes[off + 4..off + 12].try_into().unwrap());
                off += 12;
                terms.push((e, coef));
            }
            nz_terms.push(terms);
        }
        if off + 4 > bytes.len() {
            return Err("csr spec: truncated diag count".into());
        }
        let nd = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let mut diag_overrides = Vec::with_capacity(nd);
        for _ in 0..nd {
            if off + 6 > bytes.len() {
                return Err("csr spec: truncated diag".into());
            }
            let k = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let ne = u16::from_le_bytes(bytes[off + 4..off + 6].try_into().unwrap()) as usize;
            off += 6;
            let mut edges = Vec::with_capacity(ne);
            for _ in 0..ne {
                if off + 4 > bytes.len() {
                    return Err("csr spec: truncated diag edge".into());
                }
                edges.push(u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
                off += 4;
            }
            diag_overrides.push((k, edges));
        }
        Ok(Self {
            num_edges,
            nnz,
            nz_terms,
            diag_overrides,
        })
    }
}

#[cfg(all(test, feature = "sparse"))]
mod tests {
    use super::*;
    use crate::network::Network;

    #[test]
    fn spec_roundtrip_matches_fast_assemble() {
        let net = Network::arch_chain(3.0, 8, -1.0, -0.2);
        let s = Structure::from_network(&net);
        let pat = pattern_fast(&s);
        let spec = CsrAssemblySpec::from_pattern(&pat);
        let enc = spec.encode();
        let dec = CsrAssemblySpec::decode(&enc).expect("decode");
        let v0 = pat.assemble(&net.q);
        let v1 = dec.assemble(&net.q);
        let mut m = 0.0f64;
        for (a, b) in v0.iter().zip(v1.iter()) {
            m = m.max((a - b).abs());
        }
        assert!(m < 1e-12, "assemble mismatch {m}");
    }
}
