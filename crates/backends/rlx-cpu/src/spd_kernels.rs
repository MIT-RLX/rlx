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

//! CPU kernels for the core Riemannian / SPD-manifold ops (`Op::BiMap`,
//! `Op::ReEig`, `Op::LogEig`, `Op::SpdBatchNorm`, `Op::SpdKarcherMean`
//! and their backwards).
//!
//! These reuse the `CpuKernel` scaffold — the same typed-view contract
//! as user custom ops — so the core ops ride the existing
//! `Thunk::CustomOp` execution path (both the closure-compiled and the
//! match-based executors) without bespoke `Thunk` variants. The compute
//! bodies live in [`crate::spd`]; per-op parameters (`eps`/`iters`/`tol`)
//! are baked into the kernel struct at compile time (F64, CPU-first).

use crate::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};
use crate::spd;
use rlx_ir::SpdMatFn;

/// Read the `(rows, cols)` of an `[n,n]`-style square matrix input.
fn dim(t: &CpuTensorRef<'_>, i: usize) -> usize {
    t.shape().dim(i).unwrap_static()
}

/// BiMap forward: `Y = W · X · Wᵀ`. Inputs `[W [m,n], X [n,n]]`.
pub struct BiMapKernel;
impl CpuKernel for BiMapKernel {
    fn name(&self) -> &str {
        "core.bimap"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let w = inputs[0].expect_f64("bimap W")?;
        let x = inputs[1].expect_f64("bimap X")?;
        let (m, n) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let out = output.expect_f64_mut("bimap Y")?;
        out.copy_from_slice(&spd::bimap(w, x, m, n));
        Ok(())
    }
}

/// ReEig forward: `Y = U max(ε, Σ) Uᵀ`. Input `[X [n,n]]`; output is
/// the packed `[Y (n²), λ (n), U (n²)]` so the backward reuses the
/// eigendecomposition. The builder narrows out `Y`.
pub struct ReEigKernel {
    pub eps: f64,
}
impl CpuKernel for ReEigKernel {
    fn name(&self) -> &str {
        "core.reeig"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("reeig X")?;
        let n = dim(&inputs[0], 0);
        let eps = self.eps;
        let out = output.expect_f64_mut("reeig packed")?;
        out.copy_from_slice(&spd::spectral_forward_packed(x, n, |l| l.max(eps)));
        Ok(())
    }
}

/// LogEig forward: `Y = logm(X)`. Packed `[Y, λ, U]` output (see [`ReEigKernel`]).
pub struct LogEigKernel {
    pub eps: f64,
}
impl CpuKernel for LogEigKernel {
    fn name(&self) -> &str {
        "core.logeig"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("logeig X")?;
        let n = dim(&inputs[0], 0);
        let eps = self.eps;
        let out = output.expect_f64_mut("logeig packed")?;
        out.copy_from_slice(&spd::spectral_forward_packed(x, n, |l| l.max(eps).ln()));
        Ok(())
    }
}

/// SPD batch-norm transport forward. Inputs `[X [B,n,n], mean [n,n], G [n,n]]`.
pub struct SpdBatchNormKernel {
    pub eps: f64,
}
impl CpuKernel for SpdBatchNormKernel {
    fn name(&self) -> &str {
        "core.spd_batch_norm"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("spd_bn X")?;
        let mean = inputs[1].expect_f64("spd_bn mean")?;
        let g = inputs[2].expect_f64("spd_bn G")?;
        let (batch, n) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let out = output.expect_f64_mut("spd_bn Y")?;
        out.copy_from_slice(&spd::spd_bn_transport(x, mean, g, batch, n, self.eps));
        Ok(())
    }
}

/// Batch Fréchet/Karcher mean. Input `[X [B,n,n]]` → `[n,n]`.
pub struct SpdKarcherMeanKernel {
    pub iters: usize,
    pub tol: f64,
}
impl CpuKernel for SpdKarcherMeanKernel {
    fn name(&self) -> &str {
        "core.spd_karcher_mean"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("karcher X")?;
        let (batch, n) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let covs: Vec<Vec<f64>> = (0..batch)
            .map(|bi| x[bi * n * n..(bi + 1) * n * n].to_vec())
            .collect();
        let out = output.expect_f64_mut("karcher mean")?;
        out.copy_from_slice(&spd::karcher_mean(&covs, n, self.iters, self.tol));
        Ok(())
    }
}

/// Weighted batch Fréchet/Karcher mean. Inputs `[X [B,n,n], w [B]]` → `[n,n]`.
pub struct SpdKarcherMeanWeightedKernel {
    pub iters: usize,
    pub tol: f64,
}
impl CpuKernel for SpdKarcherMeanWeightedKernel {
    fn name(&self) -> &str {
        "core.spd_karcher_mean_weighted"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("karcher_w X")?;
        let w = inputs[1].expect_f64("karcher_w weights")?;
        let (batch, n) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let covs: Vec<Vec<f64>> = (0..batch)
            .map(|bi| x[bi * n * n..(bi + 1) * n * n].to_vec())
            .collect();
        let out = output.expect_f64_mut("karcher_w mean")?;
        out.copy_from_slice(&spd::karcher_mean_weighted(
            &covs, w, n, self.iters, self.tol,
        ));
        Ok(())
    }
}

/// AIRM log map at an arbitrary base. Inputs `[base [n,n], X [n,n]]` → `[n,n]`.
pub struct SpdLogMapKernel;
impl CpuKernel for SpdLogMapKernel {
    fn name(&self) -> &str {
        "core.spd_log_map"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let base = inputs[0].expect_f64("log_map base")?;
        let x = inputs[1].expect_f64("log_map X")?;
        let n = dim(&inputs[0], 0);
        let out = output.expect_f64_mut("log_map V")?;
        out.copy_from_slice(&spd::log_map(base, x, n));
        Ok(())
    }
}

/// AIRM exp map at an arbitrary base. Inputs `[base [n,n], V [n,n]]` → `[n,n]`.
pub struct SpdExpMapKernel;
impl CpuKernel for SpdExpMapKernel {
    fn name(&self) -> &str {
        "core.spd_exp_map"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let base = inputs[0].expect_f64("exp_map base")?;
        let v = inputs[1].expect_f64("exp_map V")?;
        let n = dim(&inputs[0], 0);
        let out = output.expect_f64_mut("exp_map Y")?;
        out.copy_from_slice(&spd::exp_map(base, v, n));
        Ok(())
    }
}

/// AIRM parallel transport. Inputs `[from [n,n], to [n,n], V [n,n]]` → `[n,n]`.
pub struct SpdParallelTransportKernel;
impl CpuKernel for SpdParallelTransportKernel {
    fn name(&self) -> &str {
        "core.spd_parallel_transport"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let from = inputs[0].expect_f64("transport from")?;
        let to = inputs[1].expect_f64("transport to")?;
        let v = inputs[2].expect_f64("transport V")?;
        let n = dim(&inputs[0], 0);
        let out = output.expect_f64_mut("transport out")?;
        out.copy_from_slice(&spd::parallel_transport(from, to, v, n));
        Ok(())
    }
}

/// Batched SPD spectral matrix function. Input `[X [B,n,n]]` → `[B,n,n]`.
pub struct SpdMatrixFnBatchKernel {
    pub kind: SpdMatFn,
}
impl CpuKernel for SpdMatrixFnBatchKernel {
    fn name(&self) -> &str {
        "core.spd_matrix_fn_batch"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("spd_matfn X")?;
        let (batch, n) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let covs: Vec<Vec<f64>> = (0..batch)
            .map(|bi| x[bi * n * n..(bi + 1) * n * n].to_vec())
            .collect();
        let res = match self.kind {
            SpdMatFn::Logm => spd::logm_batch(&covs, n),
            SpdMatFn::Expm => spd::expm_batch(&covs, n),
            SpdMatFn::Sqrtm => spd::sqrtm_batch(&covs, n),
            SpdMatFn::Invsqrtm => spd::invsqrtm_batch(&covs, n),
        };
        let out = output.expect_f64_mut("spd_matfn Y")?;
        out.copy_from_slice(&res.concat());
        Ok(())
    }
}

/// SpdLogMap backward. Inputs `[base [n,n], x [n,n], dY [n,n]]` →
/// packed `[d_base ∥ d_x]` `[2,n,n]`.
pub struct SpdLogMapBackwardKernel;
impl CpuKernel for SpdLogMapBackwardKernel {
    fn name(&self) -> &str {
        "core.spd_log_map_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let base = inputs[0].expect_f64("log_map_bwd base")?;
        let x = inputs[1].expect_f64("log_map_bwd x")?;
        let dy = inputs[2].expect_f64("log_map_bwd dY")?;
        let n = dim(&inputs[0], 0);
        let (d_base, d_x) = spd::log_map_backward(base, x, dy, n);
        let out = output.expect_f64_mut("log_map_bwd packed")?;
        out[..n * n].copy_from_slice(&d_base);
        out[n * n..].copy_from_slice(&d_x);
        Ok(())
    }
}

/// SpdExpMap backward. Inputs `[base [n,n], v [n,n], dY [n,n]]` →
/// packed `[d_base ∥ d_v]` `[2,n,n]`.
pub struct SpdExpMapBackwardKernel;
impl CpuKernel for SpdExpMapBackwardKernel {
    fn name(&self) -> &str {
        "core.spd_exp_map_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let base = inputs[0].expect_f64("exp_map_bwd base")?;
        let v = inputs[1].expect_f64("exp_map_bwd v")?;
        let dy = inputs[2].expect_f64("exp_map_bwd dY")?;
        let n = dim(&inputs[0], 0);
        let (d_base, d_v) = spd::exp_map_backward(base, v, dy, n);
        let out = output.expect_f64_mut("exp_map_bwd packed")?;
        out[..n * n].copy_from_slice(&d_base);
        out[n * n..].copy_from_slice(&d_v);
        Ok(())
    }
}

/// SpdParallelTransport backward. Inputs `[from, to, v, dY]` (each `[n,n]`) →
/// packed `[d_from ∥ d_to ∥ d_v]` `[3,n,n]`.
pub struct SpdParallelTransportBackwardKernel;
impl CpuKernel for SpdParallelTransportBackwardKernel {
    fn name(&self) -> &str {
        "core.spd_parallel_transport_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let from = inputs[0].expect_f64("transport_bwd from")?;
        let to = inputs[1].expect_f64("transport_bwd to")?;
        let v = inputs[2].expect_f64("transport_bwd v")?;
        let dy = inputs[3].expect_f64("transport_bwd dY")?;
        let n = dim(&inputs[0], 0);
        let (d_from, d_to, d_v) = spd::parallel_transport_backward(from, to, v, dy, n);
        let out = output.expect_f64_mut("transport_bwd packed")?;
        out[..n * n].copy_from_slice(&d_from);
        out[n * n..2 * n * n].copy_from_slice(&d_to);
        out[2 * n * n..].copy_from_slice(&d_v);
        Ok(())
    }
}

/// SpdMatrixFnBatch backward. Inputs `[X [B,n,n], dY [B,n,n]]` → `dX [B,n,n]`.
pub struct SpdMatrixFnBatchBackwardKernel {
    pub kind: SpdMatFn,
}
impl CpuKernel for SpdMatrixFnBatchBackwardKernel {
    fn name(&self) -> &str {
        "core.spd_matrix_fn_batch_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("spd_matfn_bwd X")?;
        let dy = inputs[1].expect_f64("spd_matfn_bwd dY")?;
        let n = dim(&inputs[0], 1);
        // Same (f, f') as the batch forward (see spd::{logm,expm,sqrtm,invsqrtm}_batch).
        let dx = match self.kind {
            SpdMatFn::Logm => spd::matrix_fn_batch_backward(
                x,
                dy,
                n,
                |l| l.max(1e-12).ln(),
                |l| 1.0 / l.max(1e-12),
            ),
            SpdMatFn::Expm => spd::matrix_fn_batch_backward(x, dy, n, |l| l.exp(), |l| l.exp()),
            SpdMatFn::Sqrtm => spd::matrix_fn_batch_backward(
                x,
                dy,
                n,
                |l| l.max(0.0).sqrt(),
                |l| 0.5 / l.max(1e-12).sqrt(),
            ),
            SpdMatFn::Invsqrtm => spd::matrix_fn_batch_backward(
                x,
                dy,
                n,
                |l| 1.0 / l.max(1e-12).sqrt(),
                |l| -0.5 * l.max(1e-12).powf(-1.5),
            ),
        };
        let out = output.expect_f64_mut("spd_matfn_bwd dX")?;
        out.copy_from_slice(&dx);
        Ok(())
    }
}

/// `n` from a packed `[λ ∥ U]` length `L = n²+n` (⇒ `n = (√(1+4L)−1)/2`).
fn n_from_packed(len: usize) -> usize {
    (((1 + 4 * len) as f64).sqrt().round() as usize - 1) / 2
}

/// Symmetric eigendecomposition. Input `[A n,n]` → packed `[λ ∥ U]` `[n²+n]`.
pub struct EighKernel;
impl CpuKernel for EighKernel {
    fn name(&self) -> &str {
        "core.eigh"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("eigh A")?;
        let n = dim(&inputs[0], 0);
        let out = output.expect_f64_mut("eigh packed")?;
        out.copy_from_slice(&spd::eigh_packed(a, n));
        Ok(())
    }
}

/// Eigh backward. Inputs `[fwd [n²+n], bar [n²+n]]` → `Ā [n,n]`.
pub struct EighBackwardKernel;
impl CpuKernel for EighBackwardKernel {
    fn name(&self) -> &str {
        "core.eigh_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let fwd = inputs[0].expect_f64("eigh_bwd fwd")?;
        let bar = inputs[1].expect_f64("eigh_bwd bar")?;
        let n = n_from_packed(fwd.len());
        let out = output.expect_f64_mut("eigh_bwd dA")?;
        out.copy_from_slice(&spd::eigh_backward_packed(fwd, bar, n));
        Ok(())
    }
}

/// Batched eigh. Input `[A B,n,n]` → packed `[B, n²+n]`.
pub struct EighBatchKernel;
impl CpuKernel for EighBatchKernel {
    fn name(&self) -> &str {
        "core.eigh_batch"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let a = inputs[0].expect_f64("eigh_batch A")?;
        let (b, n) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let covs: Vec<Vec<f64>> = (0..b)
            .map(|bi| a[bi * n * n..(bi + 1) * n * n].to_vec())
            .collect();
        let out = output.expect_f64_mut("eigh_batch packed")?;
        out.copy_from_slice(&spd::eigh_batch_packed(&covs, n).concat());
        Ok(())
    }
}

/// Batched eigh backward. Inputs `[fwd [B,n²+n], bar [B,n²+n]]` → `Ā [B,n,n]`.
pub struct EighBatchBackwardKernel;
impl CpuKernel for EighBatchBackwardKernel {
    fn name(&self) -> &str {
        "core.eigh_batch_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        use rayon::prelude::*;
        let fwd = inputs[0].expect_f64("eigh_batch_bwd fwd")?;
        let bar = inputs[1].expect_f64("eigh_batch_bwd bar")?;
        let (b, packed) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let n = n_from_packed(packed);
        let da: Vec<f64> = (0..b)
            .into_par_iter()
            .flat_map(|bi| {
                let f = &fwd[bi * packed..(bi + 1) * packed];
                let bb = &bar[bi * packed..(bi + 1) * packed];
                spd::eigh_backward_packed(f, bb, n)
            })
            .collect();
        let out = output.expect_f64_mut("eigh_batch_bwd dA")?;
        out.copy_from_slice(&da);
        Ok(())
    }
}

/// ReEig backward. Inputs `[λ [n], U [n²], dY [n²]]` → `dX [n,n]`,
/// reusing the forward eigendecomposition (no recompute).
pub struct ReEigBackwardKernel {
    pub eps: f64,
}
impl CpuKernel for ReEigBackwardKernel {
    fn name(&self) -> &str {
        "core.reeig_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let lam = inputs[0].expect_f64("reeig_bwd λ")?;
        let u = inputs[1].expect_f64("reeig_bwd U")?;
        let dy = inputs[2].expect_f64("reeig_bwd dY")?;
        let n = lam.len();
        let out = output.expect_f64_mut("reeig_bwd dX")?;
        out.copy_from_slice(&spd::reeig_backward_precomputed(lam, u, dy, n, self.eps));
        Ok(())
    }
}

/// LogEig backward. Inputs `[λ [n], U [n²], dY [n²]]` → `dX [n,n]`.
pub struct LogEigBackwardKernel {
    pub eps: f64,
}
impl CpuKernel for LogEigBackwardKernel {
    fn name(&self) -> &str {
        "core.logeig_backward"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let lam = inputs[0].expect_f64("logeig_bwd λ")?;
        let u = inputs[1].expect_f64("logeig_bwd U")?;
        let dy = inputs[2].expect_f64("logeig_bwd dY")?;
        let n = lam.len();
        let out = output.expect_f64_mut("logeig_bwd dX")?;
        out.copy_from_slice(&spd::logeig_backward_precomputed(lam, u, dy, n, self.eps));
        Ok(())
    }
}

/// SPD batch-norm backward w.r.t. X. Inputs `[mean [n,n], G [n,n], dY [B,n,n]]`.
pub struct SpdBnBackwardXKernel {
    pub eps: f64,
}
impl CpuKernel for SpdBnBackwardXKernel {
    fn name(&self) -> &str {
        "core.spd_batch_norm_backward_x"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let mean = inputs[0].expect_f64("spd_bn_bwd_x mean")?;
        let g = inputs[1].expect_f64("spd_bn_bwd_x G")?;
        let dy = inputs[2].expect_f64("spd_bn_bwd_x dY")?;
        let (batch, n) = (dim(&inputs[2], 0), dim(&inputs[0], 0));
        let out = output.expect_f64_mut("spd_bn_bwd_x dX")?;
        out.copy_from_slice(&spd::spd_bn_backward_x(mean, g, dy, batch, n, self.eps));
        Ok(())
    }
}

/// SPD batch-norm backward w.r.t. G. Inputs `[X [B,n,n], mean [n,n], G [n,n], dY [B,n,n]]`.
pub struct SpdBnBackwardGKernel {
    pub eps: f64,
}
impl CpuKernel for SpdBnBackwardGKernel {
    fn name(&self) -> &str {
        "core.spd_batch_norm_backward_g"
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        _attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f64("spd_bn_bwd_g X")?;
        let mean = inputs[1].expect_f64("spd_bn_bwd_g mean")?;
        let g = inputs[2].expect_f64("spd_bn_bwd_g G")?;
        let dy = inputs[3].expect_f64("spd_bn_bwd_g dY")?;
        let (batch, n) = (dim(&inputs[0], 0), dim(&inputs[0], 1));
        let out = output.expect_f64_mut("spd_bn_bwd_g dG")?;
        out.copy_from_slice(&spd::spd_bn_backward_g(x, mean, g, dy, batch, n, self.eps));
        Ok(())
    }
}
