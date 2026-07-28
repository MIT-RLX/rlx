// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! **All-backend SPD-manifold layers via the graph-Jacobi eigensolver.**
//!
//! The native SPD ops `Op::EighBatch` / `Op::SpdMatrixFnBatch` / `Op::SpdLogMap`
//! (see [`super::manifold`]) currently only carry Metal + CUDA kernels, so a
//! model that needs the AIRM manifold on CoreML / MLX cannot use them. This
//! module expresses the same operations purely through the backend-agnostic
//! graph eigensolver in [`super::spd_eig`] (cyclic Jacobi via `Op::Scan` +
//! `mm`/`sqrt`/`Activation::Log`), so they lower to **every** backend — the same
//! path the `rlx-repspd` / `rlx-mpnet` SPD-manifold ports use to reach all five
//! Mac backends (+ CUDA).
//!
//! These are additive `*_graph` variants of the manifold builders; the native
//! ops remain for CPU-f64-LAPACK / GPU-kernel fast paths where they exist.

use crate::infer::GraphExt;
use crate::op::SpdMatFn;
use crate::ops::spd_eig::{SpectralFn, spd_jacobi_sweeps, spectral_matfn_batched};
use crate::{Graph, NodeId};

impl Graph {
    /// Batched SPD spectral matrix function via the **all-backend** graph
    /// eigensolver — the portable counterpart of [`Graph::spd_matrix_fn_batch`]
    /// (which emits the Metal/CUDA-only `Op::SpdMatrixFnBatch`).
    ///
    /// `x` is `[batch, n, n]` symmetric-PSD; applies `kind` (logm / sqrtm /
    /// invsqrtm) to each matrix and returns `[batch, n, n]`. `SpdMatFn::Expm`
    /// has no Jacobi-graph form here and falls back to the native op.
    ///
    /// `eps` floors the eigenvalues (`max(λ, eps)`) exactly as
    /// [`super::spd_eig::spectral_matfn_batched`].
    pub fn spd_matrix_fn_batch_graph(&mut self, x: NodeId, kind: SpdMatFn, eps: f64) -> NodeId {
        let xs = self.node(x).shape.clone();
        let b = xs.dim(0).unwrap_static();
        let n = xs.dim(1).unwrap_static();
        let sweeps = spd_jacobi_sweeps();
        let f = match kind {
            SpdMatFn::Logm => SpectralFn::Log,
            SpdMatFn::Sqrtm => SpectralFn::Sqrt,
            SpdMatFn::Invsqrtm => SpectralFn::InvSqrt,
            // Expm has no Jacobi-graph spectral variant here — use the native op.
            SpdMatFn::Expm => return self.spd_matrix_fn_batch(x, SpdMatFn::Expm),
        };
        spectral_matfn_batched(self, x, b, n, sweeps, eps, f)
    }

    /// Batched matrix logarithm via the all-backend graph eigensolver.
    pub fn spd_logm_batch_graph(&mut self, x: NodeId, eps: f64) -> NodeId {
        self.spd_matrix_fn_batch_graph(x, SpdMatFn::Logm, eps)
    }
    /// Batched matrix square root via the all-backend graph eigensolver.
    pub fn spd_sqrtm_batch_graph(&mut self, x: NodeId, eps: f64) -> NodeId {
        self.spd_matrix_fn_batch_graph(x, SpdMatFn::Sqrtm, eps)
    }
    /// Batched inverse matrix square root via the all-backend graph eigensolver.
    pub fn spd_invsqrtm_batch_graph(&mut self, x: NodeId, eps: f64) -> NodeId {
        self.spd_matrix_fn_batch_graph(x, SpdMatFn::Invsqrtm, eps)
    }

    /// AIRM Riemannian logarithm at an arbitrary base, **all backends**:
    /// `Log_P(X) = P^{1/2} · logm(P^{-1/2} X P^{-1/2}) · P^{1/2}`.
    ///
    /// The portable counterpart of [`Graph::spd_log_map`] (native `Op::SpdLogMap`,
    /// Metal/CUDA-only). Both `base` and `x` are `[batch, n, n]` SPD; output is
    /// the `[batch, n, n]` tangent vector at `base`. This is the AIRM op the
    /// Log-Euclidean / RepSPD-style manifold cross-attention heads need on
    /// CoreML / MLX.
    pub fn spd_log_map_graph(&mut self, base: NodeId, x: NodeId, eps: f64) -> NodeId {
        let p_half = self.spd_sqrtm_batch_graph(base, eps); // P^{1/2}
        let p_ihalf = self.spd_invsqrtm_batch_graph(base, eps); // P^{-1/2}
        // inner = P^{-1/2} · X · P^{-1/2}   (batched mm)
        let px = self.mm(p_ihalf, x);
        let inner = self.mm(px, p_ihalf);
        let log_inner = self.spd_logm_batch_graph(inner, eps);
        // Log_P(X) = P^{1/2} · log_inner · P^{1/2}
        let ph_log = self.mm(p_half, log_inner);
        self.mm(ph_log, p_half)
    }

    /// AIRM Riemannian exponential at an arbitrary base, **all backends**:
    /// `Exp_P(V) = P^{1/2} · expm(P^{-1/2} V P^{-1/2}) · P^{1/2}` — inverse of
    /// [`Graph::spd_log_map_graph`]. Uses `P^{1/2}`/`P^{-1/2}` from the graph
    /// eigensolver; the inner `expm` uses the native `Op::SpdMatrixFnBatch`
    /// (`Expm` has no Jacobi-graph spectral form), so this is all-backend only
    /// where `expm` is available — prefer the log map for portability.
    pub fn spd_exp_map_graph(&mut self, base: NodeId, v: NodeId, eps: f64) -> NodeId {
        let p_half = self.spd_sqrtm_batch_graph(base, eps);
        let p_ihalf = self.spd_invsqrtm_batch_graph(base, eps);
        let pv = self.mm(p_ihalf, v);
        let inner = self.mm(pv, p_ihalf);
        let exp_inner = self.spd_matrix_fn_batch_graph(inner, SpdMatFn::Expm, eps);
        let ph_exp = self.mm(p_half, exp_inner);
        self.mm(ph_exp, p_half)
    }
}
