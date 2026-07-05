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

//! Fused vector-quantization kernel for RLX.
//!
//! `vector_quantize` in `rlx-ir` is a composition (matmul → argmin → gather)
//! that materializes the full `[N, K]` distance matrix. This crate provides a
//! **fused** nearest-codebook assignment that computes `argmin_j (‖C_j‖² −
//! 2·x·C_jᵀ)` with a running per-row reduction — never storing `[N, K]` — and
//! parallelizes over rows. Benchmarks (`rlx-runtime/tests/vq_bench.rs`) show it
//! beats the composition ~1.5–4.8×, the win growing with codebook size `K`.
//!
//! Registered as an `Op::Custom("rlx.vq_assign")` via the framework kernel
//! registries (CPU + Metal host-callback), following the same downstream
//! pattern as `rlx-linalg`. Call [`register`] once before building graphs.
//!
//! ## Device-aware lowering ([`Target`])
//!
//! VQ's core is a matmul, so the best implementation differs by backend:
//!
//! - **`Target::Cpu`** → the fused custom op (no `[N,K]` matrix, rayon over
//!   rows). Measured **1.5–5.4×** faster than the matmul+argmin composition.
//! - **`Target::Gpu`** → the on-device composition (MPS matmul + argmin). On
//!   Metal the matrix units make this **~10× faster than the CPU** path (and
//!   ~2–4× faster than a hand-written fused GPU kernel), so it is the right
//!   choice on Metal/wgpu.
//!
//! `rlx-metal` additionally recognizes `Op::Custom("rlx.vq_assign")` and
//! dispatches a native on-GPU MSL kernel (one threadgroup/row, `float4`
//! cooperative argmin), so a `Target::Cpu` graph accidentally run on Metal is
//! only ~2–4× slower than the composition, not the ~100× of the old
//! host-callback ABI. The `MetalKernel` host-callback below is a last-resort
//! portability fallback.
//!
//! ```ignore
//! rlx_vq::register();
//! let (idx, q) = rlx_vq::vector_quantize(&mut g, x, cb, Metric::L2, Target::Gpu);
//! ```

use std::sync::Arc;

use rlx_ir::{DType, Graph, NodeId, OpExtension, Shape, register_op};

/// Stable registry id, shared by the `OpExtension` and every backend kernel.
pub const VQ_ASSIGN: &str = "rlx.vq_assign";

/// Distance metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean (`argmin ‖x − C‖²`).
    L2,
    /// Cosine similarity (`argmax x·C / ‖x‖‖C‖`).
    Cosine,
}

impl Metric {
    fn tag(self) -> u8 {
        match self {
            Metric::L2 => 0,
            Metric::Cosine => 1,
        }
    }
    fn from_tag(t: u8) -> Metric {
        if t == 1 { Metric::Cosine } else { Metric::L2 }
    }
    fn to_ir(self) -> rlx_ir::ops::vq::VqMetric {
        match self {
            Metric::L2 => rlx_ir::ops::vq::VqMetric::L2,
            Metric::Cosine => rlx_ir::ops::vq::VqMetric::Cosine,
        }
    }
}

/// Which backend the graph will run on — selects the optimal VQ lowering.
///
/// VQ's core is a matmul. On CPU the fused kernel (no `[N,K]` matrix, rayon)
/// beats the matmul+argmin composition ~1.5–5.4×. On GPU the composition wins,
/// because MPS/matrix units do the matmul far faster than a hand-written kernel
/// — so `Gpu` emits the on-device composition (~10× faster than CPU).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    Cpu,
    Gpu,
}

/// Fused nearest-code assignment: inputs `x[N, D]`, `codebook[K, D]`
/// (row-major f32); returns `N` f32-encoded code indices (ready for `gather`).
///
/// L2 uses the `‖C‖² − 2·x·Cᵀ` proxy (drops the per-row-constant `‖x‖²`), the
/// same quantity the composition minimizes, so results match modulo f32
/// summation order. Parallel over rows; the inner dot auto-vectorizes.
pub fn fused_vq_assign(
    x: &[f32],
    cb: &[f32],
    n: usize,
    d: usize,
    k: usize,
    metric: Metric,
) -> Vec<f32> {
    use rayon::prelude::*;
    assert_eq!(x.len(), n * d, "vq_assign: x length");
    assert_eq!(cb.len(), k * d, "vq_assign: codebook length");

    // Per-code scalars: ‖C_j‖² for L2, 1/‖C_j‖ for cosine.
    let cb_scale: Vec<f32> = (0..k)
        .map(|j| {
            let s: f32 = cb[j * d..(j + 1) * d].iter().map(|&v| v * v).sum();
            match metric {
                Metric::L2 => s,
                Metric::Cosine => {
                    if s > 0.0 {
                        1.0 / s.sqrt()
                    } else {
                        0.0
                    }
                }
            }
        })
        .collect();

    let mut out = vec![0f32; n];
    out.par_iter_mut().enumerate().for_each(|(i, o)| {
        let xi = &x[i * d..(i + 1) * d];
        let mut best_j = 0usize;
        match metric {
            Metric::L2 => {
                let mut best = f32::INFINITY;
                for j in 0..k {
                    let cj = &cb[j * d..(j + 1) * d];
                    let mut dot = 0.0f32;
                    for t in 0..d {
                        dot += xi[t] * cj[t];
                    }
                    let dist = cb_scale[j] - 2.0 * dot; // drop ‖x‖²
                    if dist < best {
                        best = dist;
                        best_j = j;
                    }
                }
            }
            Metric::Cosine => {
                let mut best = f32::NEG_INFINITY;
                for j in 0..k {
                    let cj = &cb[j * d..(j + 1) * d];
                    let mut dot = 0.0f32;
                    for t in 0..d {
                        dot += xi[t] * cj[t];
                    }
                    let sim = dot * cb_scale[j]; // ‖x‖ constant per row → omit
                    if sim > best {
                        best = sim;
                        best_j = j;
                    }
                }
            }
        }
        *o = best_j as f32;
    });
    out
}

// ── Op extension (shape inference) ──────────────────────────────

struct VqAssignExt;
impl OpExtension for VqAssignExt {
    fn name(&self) -> &str {
        VQ_ASSIGN
    }
    fn num_inputs(&self) -> usize {
        2
    }
    fn infer_shape(&self, inputs: &[&Shape], _attrs: &[u8]) -> Shape {
        let x = inputs[0]; // [N, D]
        let cb = inputs[1]; // [K, D]
        assert_eq!(x.rank(), 2, "vq_assign: x must be [N, D]");
        assert_eq!(cb.rank(), 2, "vq_assign: codebook must be [K, D]");
        assert_eq!(
            x.dim(1).unwrap_static(),
            cb.dim(1).unwrap_static(),
            "vq_assign: x/codebook feature dim mismatch"
        );
        Shape::new(&[x.dim(0).unwrap_static()], DType::F32)
    }
}

// ── CPU kernel ──────────────────────────────────────────────────

#[cfg(feature = "cpu")]
struct VqAssignCpu;
#[cfg(feature = "cpu")]
impl rlx_cpu::op_registry::CpuKernel for VqAssignCpu {
    fn name(&self) -> &str {
        VQ_ASSIGN
    }
    fn execute(
        &self,
        inputs: &[rlx_cpu::op_registry::CpuTensorRef<'_>],
        output: rlx_cpu::op_registry::CpuTensorMut<'_>,
        attrs: &[u8],
    ) -> Result<(), String> {
        let x = inputs[0].expect_f32("vq x")?;
        let cb = inputs[1].expect_f32("vq codebook")?;
        let n = inputs[0].shape().dim(0).unwrap_static();
        let d = inputs[0].shape().dim(1).unwrap_static();
        let k = inputs[1].shape().dim(0).unwrap_static();
        let metric = Metric::from_tag(attrs.first().copied().unwrap_or(0));
        let out = output.expect_f32_mut("vq idx")?;
        out.copy_from_slice(&fused_vq_assign(x, cb, n, d, k, metric));
        Ok(())
    }
}

// ── Metal kernel (host callback over unified-memory bytes) ──────

#[cfg(all(feature = "metal", target_os = "macos"))]
#[derive(Debug)]
struct VqAssignMetal;
#[cfg(all(feature = "metal", target_os = "macos"))]
impl rlx_metal::op_registry::MetalKernel for VqAssignMetal {
    fn name(&self) -> &str {
        VQ_ASSIGN
    }
    fn execute(
        &self,
        inputs: &[(&[u8], &Shape)],
        output: (&mut [u8], &Shape),
        attrs: &[u8],
    ) -> Result<(), String> {
        let (x_bytes, x_shape) = inputs[0];
        let (cb_bytes, _cb_shape) = inputs[1];
        let n = x_shape.dim(0).unwrap_static();
        let d = x_shape.dim(1).unwrap_static();
        let x = bytes_to_f32(x_bytes);
        let cb = bytes_to_f32(cb_bytes);
        let k = cb.len() / d;
        let metric = Metric::from_tag(attrs.first().copied().unwrap_or(0));
        let idx = fused_vq_assign(&x, &cb, n, d, k, metric);
        for (dst, v) in output.0.chunks_exact_mut(4).zip(idx.iter()) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

// ── Registration + builder ──────────────────────────────────────

/// Register the fused VQ op with the framework registries (idempotent-ish —
/// re-registering replaces). Call once before building a graph that uses
/// [`vector_quantize`].
pub fn register() {
    register_op(Arc::new(VqAssignExt));
    #[cfg(feature = "cpu")]
    rlx_cpu::op_registry::register_cpu_kernel(Arc::new(VqAssignCpu));
    #[cfg(all(feature = "metal", target_os = "macos"))]
    rlx_metal::op_registry::register_metal_kernel(Arc::new(VqAssignMetal));
}

/// Nearest-codebook quantization: `x[N, D]`, `codebook[K, D]` →
/// `(indices[N], quantized[N, D])`, lowered optimally for `target` (see
/// [`Target`]). [`register`] must have been called for `Target::Cpu`.
pub fn vector_quantize(
    g: &mut Graph,
    x: NodeId,
    codebook: NodeId,
    metric: Metric,
    target: Target,
) -> (NodeId, NodeId) {
    use rlx_ir::infer::GraphExt as _;
    match target {
        // Fused custom op — beats the composition on CPU.
        Target::Cpu => {
            let idx = g.custom_op(VQ_ASSIGN, vec![metric.tag()], vec![x, codebook]);
            let quantized = g.gather_(codebook, idx, 0);
            (idx, quantized)
        }
        // On-device matmul+argmin+gather (rlx-ir composition) — MPS matrix
        // units make this the fast path on Metal/wgpu.
        Target::Gpu => g.vector_quantize(x, codebook, metric.to_ir()),
    }
}

/// Fused residual (multi-stage) VQ — the RVQ tokenizer in NeuroRVQ / BrainRVQ.
/// Quantizes `x` against `codebooks[0]`, subtracts the chosen code, quantizes
/// the residual against `codebooks[1]`, and so on. Returns the per-level
/// `indices` (one `[N]` tensor each) and the summed reconstruction `[N, D]`.
/// Each level's nearest-code search uses the fused kernel.
pub fn residual_vq(
    g: &mut Graph,
    x: NodeId,
    codebooks: &[NodeId],
    metric: Metric,
    target: Target,
) -> (Vec<NodeId>, NodeId) {
    use rlx_ir::infer::GraphExt as _;
    assert!(!codebooks.is_empty(), "residual_vq: need ≥1 codebook");
    let mut indices = Vec::with_capacity(codebooks.len());
    let (idx0, mut recon) = vector_quantize(g, x, codebooks[0], metric, target);
    indices.push(idx0);
    let mut residual = g.sub(x, recon);
    for &cb in &codebooks[1..] {
        let (idx, q) = vector_quantize(g, residual, cb, metric, target);
        indices.push(idx);
        recon = g.add(recon, q);
        residual = g.sub(residual, q);
    }
    (indices, recon)
}
