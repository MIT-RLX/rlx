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

//! Vector-quantization graph helpers — nearest-codebook assignment and
//! residual VQ, composed from primitive ops (`mm`, `sum`, `argmin`/`argmax`,
//! `gather`). Backend-agnostic: lowers entirely to existing kernels, so it
//! runs on every backend without new op lowerings.
//!
//! These are the discrete-tokenizer front-ends used across the EEG foundation
//! models in `exg` (NeuroRVQ, CodeBrain, EEGFormer, DeWave, BrainRVQ, …).
//! Today each of those does the codebook search in a host-side nested loop;
//! `vector_quantize` / `residual_vq` express the same math as one fused graph
//! subtree so it runs on-device and participates in autodiff.
//!
//! ## Distance math
//!
//! For inputs `x[N, D]` and codebook `C[K, D]`, the squared-L2 distance to
//! every code is `‖x‖² − 2·x·Cᵀ + ‖C‖²`. The `‖x‖²` term is constant across
//! the `K` codes, so `argmin` over the codes ignores it — we compute only
//! `‖C‖² − 2·x·Cᵀ` (shape `[N, K]`) and take the `argmin` along `K`.

use crate::infer::GraphExt as _;
use crate::{DType, Graph, NodeId};

/// Distance metric for [`Graph::vector_quantize`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VqMetric {
    /// Squared Euclidean distance (standard VQ-VAE codebook).
    L2,
    /// Cosine similarity (row-L2-normalized dot product, e.g. NeuroRVQ).
    Cosine,
}

impl Graph {
    /// Nearest-codebook assignment.
    ///
    /// * `x` — inputs `[N, D]`.
    /// * `codebook` — codes `[K, D]`.
    ///
    /// Returns `(indices, quantized)` where `indices` is `[N]` (f32-encoded
    /// code ids, ready to feed `gather`) and `quantized` is `[N, D]`, the
    /// selected code vectors. The straight-through estimator for training is
    /// left to the caller (`x + stop_gradient(quantized − x)`), matching the
    /// usual VQ-VAE recipe.
    pub fn vector_quantize(
        &mut self,
        x: NodeId,
        codebook: NodeId,
        metric: VqMetric,
    ) -> (NodeId, NodeId) {
        let xs = self.shape(x).clone();
        let cs = self.shape(codebook).clone();
        assert_eq!(xs.rank(), 2, "vector_quantize: x must be rank-2 [N, D]");
        assert_eq!(
            cs.rank(),
            2,
            "vector_quantize: codebook must be rank-2 [K, D]"
        );
        let d = xs.dim(1).unwrap_static();
        let k = cs.dim(0).unwrap_static();
        assert_eq!(
            cs.dim(1).unwrap_static(),
            d,
            "vector_quantize: x/codebook feature dim mismatch"
        );

        let idx = match metric {
            VqMetric::L2 => {
                // dist_proxy[N, K] = ‖C‖²(row) − 2·x·Cᵀ   (drop ‖x‖²)
                //
                // Formed as `add(‖C‖²_row, (−2)·x·Cᵀ)` rather than a `sub`: the
                // `‖C‖²_row` operand broadcasts its size-1 row axis against the
                // full `[N, K]` cross term, and `add` is commutative — so it is
                // insensitive to a backend broadcasting the operands in either
                // order (Metal's broadcast `Sub` reverses lhs/rhs).
                let cb_t = self.transpose_(codebook, vec![1, 0]); // [D, K]
                let cross = self.mm(x, cb_t); // [N, K]
                let neg_two = self.constant(-2.0, DType::F32);
                let neg_two_cross = self.mul(cross, neg_two);
                let cb_sq = self.mul(codebook, codebook); // [K, D]
                let cb_norm = self.sum(cb_sq, vec![1], false); // [K]
                let cb_norm_row = self.reshape_(cb_norm, vec![1, k as i64]); // [1, K]
                let dist = self.add(cb_norm_row, neg_two_cross); // [N, K]
                let s = crate::shape::reduce_shape(self.shape(dist), &[1], false)
                    .expect("argmin shape");
                self.argmin(dist, 1, false, s) // [N]
            }
            VqMetric::Cosine => {
                let xn = self.l2_normalize_rows(x);
                let cn = self.l2_normalize_rows(codebook);
                let cn_t = self.transpose_(cn, vec![1, 0]); // [D, K]
                let sim = self.mm(xn, cn_t); // [N, K]
                let s =
                    crate::shape::reduce_shape(self.shape(sim), &[1], false).expect("argmax shape");
                self.argmax(sim, 1, false, s) // [N]
            }
        };

        let quantized = self.gather_(codebook, idx, 0); // [N, D]
        (idx, quantized)
    }

    /// Residual (multi-stage) vector quantization.
    ///
    /// Quantizes `x` against `codebooks[0]`, subtracts the chosen code, then
    /// quantizes the residual against `codebooks[1]`, and so on. Returns the
    /// per-level `indices` (one `[N]` tensor per stage) and the summed
    /// reconstruction `quantized[N, D]` (`Σ_level code_level`). This is the
    /// RVQ tokenizer in NeuroRVQ / BrainRVQ.
    pub fn residual_vq(
        &mut self,
        x: NodeId,
        codebooks: &[NodeId],
        metric: VqMetric,
    ) -> (Vec<NodeId>, NodeId) {
        assert!(!codebooks.is_empty(), "residual_vq: need ≥1 codebook");
        let mut indices = Vec::with_capacity(codebooks.len());
        let (idx0, mut recon) = self.vector_quantize(x, codebooks[0], metric);
        indices.push(idx0);
        let mut residual = self.sub(x, recon);
        for &cb in &codebooks[1..] {
            let (idx, q) = self.vector_quantize(residual, cb, metric);
            indices.push(idx);
            recon = self.add(recon, q);
            residual = self.sub(residual, q);
        }
        (indices, recon)
    }

    /// Row-wise L2 normalization: `x / sqrt(Σ x² + eps)` over the last axis.
    fn l2_normalize_rows(&mut self, x: NodeId) -> NodeId {
        let sq = self.mul(x, x);
        let rank = self.shape(x).rank();
        let sum = self.sum(sq, vec![rank - 1], true); // [.., 1]
        let eps = self.constant(1e-12, DType::F32);
        let sum_eps = self.add(sum, eps);
        let norm = self.sqrt(sum_eps);
        self.div(x, norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Op, Shape};

    fn const_f32(g: &mut Graph, xs: &[f32], shape: &[usize]) -> NodeId {
        let mut bytes = Vec::with_capacity(xs.len() * 4);
        for x in xs {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        g.add_node(
            Op::Constant { data: bytes },
            vec![],
            Shape::new(shape, DType::F32),
        )
    }

    fn dims(g: &Graph, id: NodeId) -> Vec<usize> {
        g.shape(id)
            .dims()
            .iter()
            .map(|d| d.unwrap_static())
            .collect()
    }

    #[test]
    fn vq_shapes_l2() {
        let mut g = Graph::new("vq");
        let x = const_f32(&mut g, &[0.0; 3 * 4], &[3, 4]);
        let cb = const_f32(&mut g, &[0.0; 5 * 4], &[5, 4]);
        let (idx, q) = g.vector_quantize(x, cb, VqMetric::L2);
        assert_eq!(dims(&g, idx), vec![3]);
        assert_eq!(dims(&g, q), vec![3, 4]);
    }

    #[test]
    fn vq_shapes_cosine() {
        let mut g = Graph::new("vq_cos");
        let x = const_f32(&mut g, &[0.0; 2 * 6], &[2, 6]);
        let cb = const_f32(&mut g, &[0.0; 8 * 6], &[8, 6]);
        let (idx, q) = g.vector_quantize(x, cb, VqMetric::Cosine);
        assert_eq!(dims(&g, idx), vec![2]);
        assert_eq!(dims(&g, q), vec![2, 6]);
    }

    #[test]
    fn rvq_shapes() {
        let mut g = Graph::new("rvq");
        let x = const_f32(&mut g, &[0.0; 4 * 4], &[4, 4]);
        let cbs: Vec<_> = (0..3)
            .map(|_| const_f32(&mut g, &[0.0; 16 * 4], &[16, 4]))
            .collect();
        let (indices, recon) = g.residual_vq(x, &cbs, VqMetric::L2);
        assert_eq!(indices.len(), 3);
        assert_eq!(dims(&g, recon), vec![4, 4]);
    }
}
