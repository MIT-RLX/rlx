// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batched, differentiable SPD spectral functions on the CPU — **ReEig**, **LogEig**,
//! `√`, `1/√` and powers over a `[B,n,n]` stack in one kernel call.
//!
//! The portable path ([`rlx_ir::ops::spd_eig::spectral_matfn_batched`]) runs a
//! cyclic Jacobi eigensolver as an `Op::Scan` of graph primitives so it works
//! on every backend. On the CPU that is the slow way: each rotation is a
//! separate scan step on tensors far below any parallel threshold, and
//! training differentiates through every one of them. TensorCSPNet's sweep
//! time is dominated by it.
//!
//! Here each matrix gets one LAPACK `dsyevd` in f64, the batch runs
//! rayon-parallel, and the backward is the Daleckii–Krein (Loewner) adjoint
//! reusing the forward's eigendecomposition — the same math as the native
//! single-matrix `Op::ReEig`/`Op::LogEig` kernels ([`crate::spd`]), batched
//! and for f32 graphs too (computed in f64, stored in the graph's dtype).
//!
//! Two custom ops, registered on first use by [`spectral_batched`]:
//! `spd.matfn_batched` (`X [B,n,n]` → packed `[B, 2n²+n]` = `Y ‖ λ ‖ U` per
//! matrix) and `spd.matfn_batched_bwd` (`packed, d(packed)` → `dX [B,n,n]`).
//! They have no lowering rule: a graph that uses them runs on the CPU
//! backend only. Callers pick this path for the CPU and the Jacobi builders
//! for everything else.

use std::sync::{Arc, Once};

use rayon::prelude::*;
use rlx_ir::{DType, Graph, GraphExt, Node, NodeId, Op, OpExtension, Shape, VjpContext};

use crate::op_registry::{CpuKernel, CpuTensorMut, CpuTensorRef};
use crate::spd;

const FWD: &str = "spd.matfn_batched";
const BWD: &str = "spd.matfn_batched_bwd";

/// Which spectral function, `f(λ)` applied as `U·diag(f(λ))·Uᵀ` with
/// `m = max(λ, ε)` — the same floors as the graph Jacobi builders
/// (`spd_*_batched`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BatchedSpectral {
    /// `m` — SPDNet's ReEig.
    Re,
    /// `log m` — SPDNet's LogEig.
    Log,
    /// `√m`.
    Sqrt,
    /// `1/√m`.
    InvSqrt,
    /// `mᵖ` — the power deformation of the generalized Bures–Wasserstein metric.
    Pow(f64),
}

impl BatchedSpectral {
    /// `(f(λ), f′(λ))` with the floor `ε`; `f′ = 0` where the floor clamps —
    /// the derivative of `f(max(λ, ε))`, as the graph Jacobi builders compute
    /// it. (For LogEig this differs from the native single-matrix
    /// `Op::LogEig`, whose `1/m` treats `ε` as a safety floor only; callers here
    /// use `Log` with `ε = 1e-4` as ReEig∘LogEig folded, where a clamped
    /// eigenvalue must pass no gradient.)
    fn eval(self, l: f64, eps: f64) -> (f64, f64) {
        let m = l.max(eps);
        let live = l > eps;
        match self {
            Self::Re => (m, if live { 1.0 } else { 0.0 }),
            Self::Log => (m.ln(), if live { 1.0 / m } else { 0.0 }),
            Self::Sqrt => (m.sqrt(), if live { 0.5 / m.sqrt() } else { 0.0 }),
            Self::InvSqrt => (
                1.0 / m.sqrt(),
                if live { -0.5 / (m * m.sqrt()) } else { 0.0 },
            ),
            Self::Pow(p) => (m.powf(p), if live { p * m.powf(p - 1.0) } else { 0.0 }),
        }
    }
}

fn encode(kind: BatchedSpectral, eps: f64) -> Vec<u8> {
    let (k, p) = match kind {
        BatchedSpectral::Re => (0u8, 0.0),
        BatchedSpectral::Log => (1, 0.0),
        BatchedSpectral::Sqrt => (2, 0.0),
        BatchedSpectral::InvSqrt => (3, 0.0),
        BatchedSpectral::Pow(p) => (4, p),
    };
    let mut a = vec![k];
    a.extend_from_slice(&eps.to_le_bytes());
    a.extend_from_slice(&p.to_le_bytes());
    a
}

fn decode(attrs: &[u8]) -> Result<(BatchedSpectral, f64), String> {
    if attrs.len() != 17 {
        return Err(format!(
            "{FWD}: attrs must be 17 bytes, got {}",
            attrs.len()
        ));
    }
    let p = f64::from_le_bytes(attrs[9..17].try_into().unwrap());
    let kind = match attrs[0] {
        0 => BatchedSpectral::Re,
        1 => BatchedSpectral::Log,
        2 => BatchedSpectral::Sqrt,
        3 => BatchedSpectral::InvSqrt,
        4 => BatchedSpectral::Pow(p),
        k => return Err(format!("{FWD}: unknown kind {k}")),
    };
    Ok((kind, f64::from_le_bytes(attrs[1..9].try_into().unwrap())))
}

fn custom_attrs(node: &Node) -> &[u8] {
    match &node.op {
        Op::Custom { attrs, .. } => attrs,
        _ => &[],
    }
}

/// `n` from a packed row length `2n² + n`.
fn n_of_packed(len: usize) -> usize {
    // n = (−1 + √(1 + 8·len)) / 4
    let n = ((((1 + 8 * len) as f64).sqrt() - 1.0) / 4.0).round() as usize;
    assert_eq!(2 * n * n + n, len, "packed spectral row of length {len}");
    n
}

struct FwdExt;
impl OpExtension for FwdExt {
    fn name(&self) -> &str {
        FWD
    }
    fn num_inputs(&self) -> usize {
        1
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let s = inputs[0];
        let (b, n) = (s.dim(0).unwrap_static(), s.dim(1).unwrap_static());
        Shape::new(&[b, 2 * n * n + n], s.dtype())
    }
    fn vjp(&self, node: &Node, ctx: &mut VjpContext) -> Vec<(usize, NodeId)> {
        let packed = ctx.fwd_map[&node.id];
        let dx = ctx
            .bwd
            .custom_op(BWD, custom_attrs(node).to_vec(), vec![packed, ctx.upstream]);
        vec![(0, dx)]
    }
    fn verify(&self, _node: &Node, inputs: &[&Shape]) -> Vec<String> {
        let s = inputs[0];
        let mut e = Vec::new();
        if s.rank() != 3 || s.dim(1) != s.dim(2) {
            e.push(format!("{FWD}: input must be [B,n,n], got {s:?}"));
        }
        if !matches!(s.dtype(), DType::F32 | DType::F64) {
            e.push(format!("{FWD}: f32 or f64 only, got {:?}", s.dtype()));
        }
        e
    }
}

struct BwdExt;
impl OpExtension for BwdExt {
    fn name(&self) -> &str {
        BWD
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let s = inputs[0];
        let (b, n) = (
            s.dim(0).unwrap_static(),
            n_of_packed(s.dim(1).unwrap_static()),
        );
        Shape::new(&[b, n, n], s.dtype())
    }
}

/// Run `f` per matrix in f64 over a `[B, in_len]` input, writing `[B, out_len]`.
fn per_matrix<T: Copy + Send + Sync>(
    inputs: &[&[T]],
    out: &mut [T],
    in_lens: &[usize],
    out_len: usize,
    to: impl Fn(T) -> f64 + Sync,
    from: impl Fn(f64) -> T + Sync,
    f: impl Fn(&[Vec<f64>]) -> Vec<f64> + Sync,
) {
    out.par_chunks_mut(out_len).enumerate().for_each(|(b, o)| {
        let args: Vec<Vec<f64>> = inputs
            .iter()
            .zip(in_lens)
            .map(|(x, &l)| x[b * l..(b + 1) * l].iter().map(|&v| to(v)).collect())
            .collect();
        for (dst, v) in o.iter_mut().zip(f(&args)) {
            *dst = from(v);
        }
    });
}

struct FwdKernel;
impl CpuKernel for FwdKernel {
    fn name(&self) -> &str {
        FWD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let (kind, eps) = decode(attrs)?;
        let n = inputs[0].shape().dim(1).unwrap_static();
        let fwd =
            move |a: &[Vec<f64>]| spd::spectral_forward_packed(&a[0], n, |l| kind.eval(l, eps).0);
        let (il, ol) = (n * n, 2 * n * n + n);
        match inputs[0] {
            CpuTensorRef::F32 { data, .. } => per_matrix(
                &[data],
                output.expect_f32_mut(FWD)?,
                &[il],
                ol,
                |v| v as f64,
                |v| v as f32,
                fwd,
            ),
            CpuTensorRef::F64 { data, .. } => per_matrix(
                &[data],
                output.expect_f64_mut(FWD)?,
                &[il],
                ol,
                |v| v,
                |v| v,
                fwd,
            ),
            ref other => return Err(format!("{FWD}: f32 or f64 only, got {:?}", other.dtype())),
        }
        Ok(())
    }
}

struct BwdKernel;
impl CpuKernel for BwdKernel {
    fn name(&self) -> &str {
        BWD
    }
    fn execute(
        &self,
        inputs: &[CpuTensorRef<'_>],
        output: CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let (kind, eps) = decode(attrs)?;
        let pl = inputs[0].shape().dim(1).unwrap_static();
        let n = n_of_packed(pl);
        let nn = n * n;
        // args[0] = packed (Y ‖ λ ‖ U), args[1] = its cotangent; only the Y part
        // of the cotangent is read — λ and U are internal to the op.
        let bwd = move |a: &[Vec<f64>]| {
            let (lam, u) = (&a[0][nn..nn + n], &a[0][nn + n..]);
            let dy = &a[1][..nn];
            match kind {
                // ReEig: the native single-matrix kernel, exactly
                BatchedSpectral::Re => spd::reeig_backward_precomputed(lam, u, dy, n, eps),
                _ => spd::spectral_backward_precomputed(
                    lam,
                    u,
                    dy,
                    n,
                    |l| kind.eval(l, eps).0,
                    |l| kind.eval(l, eps).1,
                ),
            }
        };
        match (&inputs[0], &inputs[1]) {
            (CpuTensorRef::F32 { data: p, .. }, CpuTensorRef::F32 { data: g, .. }) => per_matrix(
                &[p, g],
                output.expect_f32_mut(BWD)?,
                &[pl, pl],
                nn,
                |v| v as f64,
                |v| v as f32,
                bwd,
            ),
            (CpuTensorRef::F64 { data: p, .. }, CpuTensorRef::F64 { data: g, .. }) => per_matrix(
                &[p, g],
                output.expect_f64_mut(BWD)?,
                &[pl, pl],
                nn,
                |v| v,
                |v| v,
                bwd,
            ),
            (a, b) => {
                return Err(format!(
                    "{BWD}: f32 or f64 only, got {:?} / {:?}",
                    a.dtype(),
                    b.dtype()
                ));
            }
        }
        Ok(())
    }
}

/// Register both ops and their CPU kernels (idempotent).
pub fn register() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        rlx_ir::register_op(Arc::new(FwdExt));
        rlx_ir::register_op(Arc::new(BwdExt));
        crate::op_registry::register_cpu_kernel(Arc::new(FwdKernel));
        crate::op_registry::register_cpu_kernel(Arc::new(BwdKernel));
    });
}

/// `U·diag(f(λ))·Uᵀ` for each matrix of `x: [batch,n,n]` (f32 or f64),
/// differentiable, CPU backend only. Returns `[batch,n,n]`.
pub fn spectral_batched(
    g: &mut Graph,
    x: NodeId,
    batch: usize,
    n: usize,
    eps: f64,
    kind: BatchedSpectral,
) -> NodeId {
    register();
    let packed = g.custom_op(FWD, encode(kind, eps), vec![x]);
    let y = g.narrow_(packed, 1, 0, n * n);
    g.reshape_(y, vec![batch as i64, n as i64, n as i64])
}

/// [`spectral_batched`] as a `Graph` method, the drop-in for the Jacobi
/// builders: `g.spd_reeig_batched(x, b, n, sweeps, eps)` becomes
/// `g.spd_native(x, b, n, eps, BatchedSpectral::Re)`.
pub trait SpectralBatchedExt {
    fn spd_native(
        &mut self,
        x: NodeId,
        batch: usize,
        n: usize,
        eps: f64,
        kind: BatchedSpectral,
    ) -> NodeId;
}

impl SpectralBatchedExt for Graph {
    fn spd_native(
        &mut self,
        x: NodeId,
        batch: usize,
        n: usize,
        eps: f64,
        kind: BatchedSpectral,
    ) -> NodeId {
        spectral_batched(self, x, batch, n, eps, kind)
    }
}
