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

//! Riemannian / SPD-manifold layer builders — SPDNet (BiMap / ReEig /
//! LogEig) and SPD batch-norm.
//!
//! These operate on symmetric-positive-definite matrices (covariance
//! descriptors) and are the geometric-deep-learning counterparts of
//! Linear / ReLU / flatten / batch-norm. CPU-first (F64); see
//! `rlx_cpu::spd` for the compute bodies and `Op::BiMap` &co for the
//! math.

use crate::{Graph, NodeId, Op, Shape, SpdMatFn};

impl Graph {
    /// BiMap (bilinear mapping) SPDNet layer: `Y = W · X · Wᵀ`.
    /// `w` is `[m, n]` (semi-orthogonal, constrained by the optimizer);
    /// `x` is `[n, n]` symmetric SPD. Output `Y` is `[m, m]`.
    pub fn bimap(&mut self, w: NodeId, x: NodeId) -> NodeId {
        let ws = self.node(w).shape.clone();
        let m = ws.dim(0).unwrap_static();
        let dtype = ws.dtype();
        self.push(Op::BiMap, vec![w, x], Shape::new(&[m, m], dtype), None)
    }

    /// ReEig (eigenvalue rectification) SPDNet nonlinearity:
    /// `Y = U · max(ε, Σ) · Uᵀ`. `x` is `[n, n]` symmetric SPD; output
    /// `Y` is the same shape. The SPD analogue of ReLU.
    ///
    /// The underlying op emits a packed `[Y, λ, U]` buffer (so the
    /// backward reuses the eigendecomposition); this builder narrows out
    /// the `[n, n]` `Y` view.
    pub fn reeig(&mut self, x: NodeId, eps: f32) -> NodeId {
        self.spectral_layer(Op::ReEig { eps }, x)
    }

    /// LogEig SPDNet layer: `Y = logm(X) = U · log(Σ) · Uᵀ`. Maps the
    /// SPD manifold to the tangent space at the identity. `x` is
    /// `[n, n]`; output `Y` is the same shape (packed-buffer detail as in
    /// [`Graph::reeig`]).
    pub fn logeig(&mut self, x: NodeId, eps: f32) -> NodeId {
        self.spectral_layer(Op::LogEig { eps }, x)
    }

    /// Shared builder for the packed spectral layers (ReEig / LogEig):
    /// push the op (output `[2n²+n]` = `Y ∥ λ ∥ U`), then narrow +
    /// reshape the leading `Y` block to `[n, n]`.
    fn spectral_layer(&mut self, op: Op, x: NodeId) -> NodeId {
        let xs = self.node(x).shape.clone();
        let n = xs.dim(0).unwrap_static();
        let dtype = xs.dtype();
        let packed = self.push(op, vec![x], Shape::new(&[2 * n * n + n], dtype), None);
        let y_flat = self.add_node(
            Op::Narrow {
                axis: 0,
                start: 0,
                len: n * n,
            },
            vec![packed],
            Shape::new(&[n * n], dtype),
        );
        self.add_node(
            Op::Reshape {
                new_shape: vec![n as i64, n as i64],
            },
            vec![y_flat],
            Shape::new(&[n, n], dtype),
        )
    }

    /// Batch Fréchet / Karcher mean of a stack of SPD matrices under
    /// the AIRM metric. `x` is `[batch, n, n]`; output is `[n, n]`.
    /// Non-differentiable (a detached batch statistic).
    pub fn spd_karcher_mean(&mut self, x: NodeId, iters: u32, tol: f32) -> NodeId {
        let xs = self.node(x).shape.clone();
        let n = xs.dim(1).unwrap_static();
        self.push(
            Op::SpdKarcherMean { iters, tol },
            vec![x],
            Shape::new(&[n, n], xs.dtype()),
            None,
        )
    }

    /// Weighted batch Fréchet / Karcher mean under the AIRM metric — the
    /// barycentre `argmin_M Σᵢ wᵢ δ²(M, Cᵢ)`. `x` is `[batch, n, n]`,
    /// `weights` is `[batch]`; output is `[n, n]`. The weighted counterpart of
    /// [`Graph::spd_karcher_mean`] (the true AIRM barycentre a barycentric-OT
    /// projection / weighted-MDM / soft-clustering needs). Non-differentiable
    /// (a detached prototype).
    pub fn spd_karcher_mean_weighted(
        &mut self,
        x: NodeId,
        weights: NodeId,
        iters: u32,
        tol: f32,
    ) -> NodeId {
        let xs = self.node(x).shape.clone();
        let n = xs.dim(1).unwrap_static();
        self.push(
            Op::SpdKarcherMeanWeighted { iters, tol },
            vec![x, weights],
            Shape::new(&[n, n], xs.dtype()),
            None,
        )
    }

    /// AIRM Riemannian logarithm at an arbitrary base point:
    /// `Log_P(X) = P^{1/2} logm(P^{-1/2} X P^{-1/2}) P^{1/2}`. `base` and `x`
    /// are `[n, n]` SPD; output is the `[n, n]` tangent vector at `base`.
    pub fn spd_log_map(&mut self, base: NodeId, x: NodeId) -> NodeId {
        let shape = self.node(x).shape.clone();
        self.push(Op::SpdLogMap, vec![base, x], shape, None)
    }

    /// AIRM Riemannian exponential at an arbitrary base point:
    /// `Exp_P(V) = P^{1/2} expm(P^{-1/2} V P^{-1/2}) P^{1/2}`. `base` is
    /// `[n, n]` SPD, `v` is a `[n, n]` symmetric tangent vector; output is the
    /// `[n, n]` SPD point. Inverse of [`Graph::spd_log_map`].
    pub fn spd_exp_map(&mut self, base: NodeId, v: NodeId) -> NodeId {
        let shape = self.node(v).shape.clone();
        self.push(Op::SpdExpMap, vec![base, v], shape, None)
    }

    /// AIRM parallel transport of a tangent vector between base points:
    /// `Γ_{P→Q}(V) = E V Eᵀ`, `E = (Q P^{-1})^{1/2}`. `from`/`to` are `[n, n]`
    /// SPD, `v` is a `[n, n]` symmetric tangent vector at `from`; output is the
    /// transported `[n, n]` tangent vector at `to`.
    pub fn spd_parallel_transport(&mut self, from: NodeId, to: NodeId, v: NodeId) -> NodeId {
        let shape = self.node(v).shape.clone();
        self.push(Op::SpdParallelTransport, vec![from, to, v], shape, None)
    }

    /// Batched SPD spectral matrix function — applies `kind`
    /// (logm / expm / sqrtm / invsqrtm) to each matrix of `x` `[batch, n, n]`;
    /// output matches `x`. The graph-level, any-backend counterpart of
    /// `rlx_cpu::spd::{logm,expm,sqrtm,invsqrtm}_batch`; see the
    /// [`Graph::spd_logm_batch`] &co convenience wrappers.
    pub fn spd_matrix_fn_batch(&mut self, x: NodeId, kind: SpdMatFn) -> NodeId {
        let shape = self.node(x).shape.clone();
        self.push(Op::SpdMatrixFnBatch { kind }, vec![x], shape, None)
    }

    /// Batched matrix logarithm — [`Graph::spd_matrix_fn_batch`] with [`SpdMatFn::Logm`].
    pub fn spd_logm_batch(&mut self, x: NodeId) -> NodeId {
        self.spd_matrix_fn_batch(x, SpdMatFn::Logm)
    }
    /// Batched matrix exponential — [`Graph::spd_matrix_fn_batch`] with [`SpdMatFn::Expm`].
    pub fn spd_expm_batch(&mut self, x: NodeId) -> NodeId {
        self.spd_matrix_fn_batch(x, SpdMatFn::Expm)
    }
    /// Batched matrix square root — [`Graph::spd_matrix_fn_batch`] with [`SpdMatFn::Sqrtm`].
    pub fn spd_sqrtm_batch(&mut self, x: NodeId) -> NodeId {
        self.spd_matrix_fn_batch(x, SpdMatFn::Sqrtm)
    }
    /// Batched inverse matrix square root — [`Graph::spd_matrix_fn_batch`] with [`SpdMatFn::Invsqrtm`].
    pub fn spd_invsqrtm_batch(&mut self, x: NodeId) -> NodeId {
        self.spd_matrix_fn_batch(x, SpdMatFn::Invsqrtm)
    }

    /// Symmetric **eigendecomposition** `A = U diag(λ) Uᵀ`. `x` is `[n, n]`
    /// (symmetric); returns `(λ [n]` ascending, `U [n, n]` with column `j` =
    /// eigenvector `j)`. Differentiable (full symmetric-eigendecomposition
    /// adjoint). Emits [`Op::Eigh`] (packed `[λ ∥ U]`) and narrows the two views.
    pub fn eigh(&mut self, x: NodeId) -> (NodeId, NodeId) {
        let xs = self.node(x).shape.clone();
        let n = xs.dim(0).unwrap_static();
        let dtype = xs.dtype();
        let packed = self.push(Op::Eigh, vec![x], Shape::new(&[n * n + n], dtype), None);
        let lam = self.add_node(
            Op::Narrow {
                axis: 0,
                start: 0,
                len: n,
            },
            vec![packed],
            Shape::new(&[n], dtype),
        );
        let u_flat = self.add_node(
            Op::Narrow {
                axis: 0,
                start: n,
                len: n * n,
            },
            vec![packed],
            Shape::new(&[n * n], dtype),
        );
        let u = self.add_node(
            Op::Reshape {
                new_shape: vec![n as i64, n as i64],
            },
            vec![u_flat],
            Shape::new(&[n, n], dtype),
        );
        (lam, u)
    }

    /// Batched symmetric eigendecomposition. `x` is `[batch, n, n]`; returns
    /// `(λ [batch, n], U [batch, n, n])`. Emits [`Op::EighBatch`].
    pub fn eigh_batch(&mut self, x: NodeId) -> (NodeId, NodeId) {
        let xs = self.node(x).shape.clone();
        let b = xs.dim(0).unwrap_static();
        let n = xs.dim(1).unwrap_static();
        let dtype = xs.dtype();
        let packed = self.push(
            Op::EighBatch,
            vec![x],
            Shape::new(&[b, n * n + n], dtype),
            None,
        );
        let lam = self.add_node(
            Op::Narrow {
                axis: 1,
                start: 0,
                len: n,
            },
            vec![packed],
            Shape::new(&[b, n], dtype),
        );
        let u_flat = self.add_node(
            Op::Narrow {
                axis: 1,
                start: n,
                len: n * n,
            },
            vec![packed],
            Shape::new(&[b, n * n], dtype),
        );
        let u = self.add_node(
            Op::Reshape {
                new_shape: vec![b as i64, n as i64, n as i64],
            },
            vec![u_flat],
            Shape::new(&[b, n, n], dtype),
        );
        (lam, u)
    }

    /// SPD batch-norm affine transport (raw op):
    /// `Y_i = G^{1/2} (M^{-1/2} X_i M^{-1/2}) G^{1/2}`.
    /// `x` is `[batch, n, n]`, `mean` (detached) and `g` (learnable) are
    /// `[n, n]`. Output matches `x`. `mean` receives no gradient; grads
    /// flow to `x` and `g`. See [`Graph::spd_batch_norm`] for the
    /// training-mode wrapper that computes `mean` from the batch.
    pub fn spd_batch_norm_transport(
        &mut self,
        x: NodeId,
        mean: NodeId,
        g: NodeId,
        eps: f32,
    ) -> NodeId {
        let shape = self.node(x).shape.clone();
        self.push(Op::SpdBatchNorm { eps }, vec![x, mean, g], shape, None)
    }

    /// Training-mode SPD batch-norm: computes the batch Fréchet mean,
    /// centers + biases by the learnable SPD `g`, and returns
    /// `(y, batch_mean)`. The caller updates the running mean out of band
    /// (e.g. `rlx_cpu::spd::geodesic_interp(running, batch_mean, momentum)`)
    /// — a non-differentiable buffer, exactly like Euclidean batch-norm.
    /// For inference, call [`Graph::spd_batch_norm_transport`] directly
    /// with the stored running mean.
    ///
    /// `x`: `[batch, n, n]`, `g`: `[n, n]`. `iters`/`tol` bound the
    /// Karcher-mean iteration.
    pub fn spd_batch_norm(
        &mut self,
        x: NodeId,
        g: NodeId,
        eps: f32,
        iters: u32,
        tol: f32,
    ) -> (NodeId, NodeId) {
        let mean = self.spd_karcher_mean(x, iters, tol);
        let y = self.spd_batch_norm_transport(x, mean, g, eps);
        (y, mean)
    }
}
